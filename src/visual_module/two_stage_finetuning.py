"""
two_stage_finetuning.py
========================
Helper utilities for the two-stage fine-tuning notebook.

Stage 1 – fine-tune on a small streamed subset of nebula/GenImage-arrow
Stage 2 – continue fine-tuning on julienlucas/midjourney-dalle-sd-nanobananapro-dataset

This module is imported by the notebook; all heavy lifting lives here so
the notebook stays beginner-friendly.
"""

import os, json, io, torch, numpy as np
from PIL import Image
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# ── helpers ────────────────────────────────────────────────────────────────────

def get_device():
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(eval_pred):
    """Metric function used by HuggingFace Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# ── Layer freezing ─────────────────────────────────────────────────────────

def freeze_encoder_layers(model, num_layers_to_freeze=10):
    """
    Freeze the first N encoder layers of a ViT model.

    This preserves the model's general visual features (edges, textures,
    shapes) while allowing the last few layers and the classifier head to
    adapt to AI-vs-real classification.

    For the dima806 ViT with 12 encoder layers:
      freeze=10 → last 2 layers + head trainable (~14 M params)
      freeze=8  → last 4 layers + head trainable (~28 M params)
    """
    # Freeze patch + position embeddings
    for param in model.vit.embeddings.parameters():
        param.requires_grad = False

    # Freeze the first N transformer blocks
    total_layers = len(model.vit.encoder.layer)
    n = min(num_layers_to_freeze, total_layers)
    for i in range(n):
        for param in model.vit.encoder.layer[i].parameters():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  🧊  Froze {n}/{total_layers} encoder layers + embeddings")
    print(f"      Trainable: {trainable:,} / {total:,} params "
          f"({100 * trainable / total:.1f}%)")


# ── Model loader ───────────────────────────────────────────────────────────────

BASE_MODEL_ID = "dima806/ai_vs_human_generated_image_detection"


def load_model_from(source="base", device=None):
    """
    Load a model + processor from any source, ready for training or evaluation.

    Parameters
    ----------
    source : str
        One of:
        - ``"base"`` → load the original dima806 HuggingFace model
        - A local directory path (e.g. ``"outputs/models/run_01_genimage"``)
        - A HuggingFace model ID
    device : torch.device or None
        Device to place the model on.  Auto-detected if None.

    Returns
    -------
    (model, processor)
        The model has a ``nn.Sequential(Dropout(0.3), Linear)`` classifier head
        attached for consistency across all training runs.
    """
    if device is None:
        device = get_device()

    model_id = BASE_MODEL_ID if source == "base" else source

    print(f"📦  Loading model from: {model_id}")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(
        model_id, ignore_mismatched_sizes=True
    ).to(device)


    n_params = sum(p.numel() for p in model.parameters())
    print(f"✅  Model loaded  –  {n_params:,} parameters  (device: {device})")
    return model, processor


# ── GenImage label extraction ──────────────────────────────────────────────────

def extract_genimage_label(example):
    """
    Derive a binary label from the GenImage `image_path` field.

    The GenImage dataset encodes the class in the file path:
      - paths containing '/ai/' → label 1  (AI-generated)
      - paths containing '/nature/' or '/real/' → label 0  (real / natural photo)

    This mirrors the dima806 model convention: 0 = human, 1 = AI-generated.
    """
    path = (example.get("image_path") or "").lower()
    if "/nature/" in path or "/real/" in path:
        example["label"] = 0
    else:
        example["label"] = 1
    return example


# ── Streaming subset builder ──────────────────────────────────────────────────

def build_genimage_subset(
    train_per_class=1000,
    val_per_class=250,
    test_per_class=500,
    seed=42,
    hf_token=None,
):
    """
    Stream nebula/GenImage-arrow and collect a small balanced subset.

    We never download the full dataset – we iterate the stream and stop once
    we have enough samples per class for each split.

    Returns
    -------
    DatasetDict  with splits 'train', 'validation', 'test'
    """
    print("📡  Streaming GenImage-arrow (this may take a few minutes)…")

    kwargs = {"streaming": True, "split": "train"}
    if hf_token:
        kwargs["token"] = hf_token
    stream = load_dataset("nebula/GenImage-arrow", **kwargs)
    stream = stream.shuffle(seed=seed, buffer_size=5000)

    # ── peek at the first record so you can see the columns ──
    first = None
    for ex in stream:
        first = ex
        break
    if first is None:
        raise RuntimeError("GenImage stream returned no records.")
    print(f"   Columns: {list(first.keys())}")
    print(f"   Example image_path: {first.get('image_path', 'N/A')}")

    # ── quota per split ──
    need = {
        "train":      {"real": train_per_class, "ai": train_per_class},
        "validation": {"real": val_per_class,   "ai": val_per_class},
        "test":       {"real": test_per_class,  "ai": test_per_class},
    }
    total_needed = sum(v for d in need.values() for v in d.values())

    buckets = {split: {"real": [], "ai": []} for split in need}
    collected = 0

    # Re-open the stream (iterators are single-pass)
    stream = load_dataset("nebula/GenImage-arrow", **kwargs)
    stream = stream.shuffle(seed=seed, buffer_size=5000)

    for example in stream:
        example = extract_genimage_label(example)
        cls = "real" if example["label"] == 0 else "ai"

        for split in ("train", "validation", "test"):
            if len(buckets[split][cls]) < need[split][cls]:
                buckets[split][cls].append(example)
                collected += 1
                break

        if collected % 500 == 0:
            print(f"   collected {collected}/{total_needed}…")
        if collected >= total_needed:
            break

    # ── assemble HF Dataset objects ──
    splits = {}
    for split in ("train", "validation", "test"):
        rows = buckets[split]["real"] + buckets[split]["ai"]
        np.random.seed(seed)
        np.random.shuffle(rows)
        splits[split] = Dataset.from_list(rows)
        print(f"   {split}: {len(splits[split])} samples "
              f"(real={len(buckets[split]['real'])}, ai={len(buckets[split]['ai'])})")

    return DatasetDict(splits)


# ── julienlucas dataset loader ────────────────────────────────────────────────

def load_julienlucas_dataset(val_ratio=0.1, seed=42, hf_token=None):
    """
    Load julienlucas/midjourney-dalle-sd-nanobananapro-dataset.

    The dataset already has train/test splits with a 'label' column
    (0 = real, 1 = AI-generated) — matching the dima806 convention.

    If no validation split exists we carve one from training data.
    """
    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token
    ds = load_dataset(
        "julienlucas/midjourney-dalle-sd-nanobananapro-dataset", **kwargs
    )
    print(f"✅  julienlucas dataset loaded  –  splits: {list(ds.keys())}")
    for s in ds:
        print(f"   {s}: {len(ds[s])} samples")

    if "validation" not in ds:
        split = ds["train"].train_test_split(test_size=val_ratio, seed=seed)
        ds = DatasetDict({
            "train": split["train"],
            "validation": split["test"],
            "test": ds["test"],
        })
        print(f"   → created validation split ({len(ds['validation'])} samples)")

    # Confirm label column
    sample = ds["train"][0]
    print(f"   Label column present: {'label' in sample}")
    print(f"   Sample label value : {sample.get('label')}")
    return ds


# ── Preprocessing (shared for both datasets) ──────────────────────────────────

def make_transform(processor):
    """
    Return a batched transform function compatible with HF Trainer.
    Converts images to RGB and applies the model's image processor.
    """
    def _transform(examples):
        images = examples["image"]
        if not isinstance(images, list):
            images = [images]
        # Handle raw bytes (e.g. GenImage) as well as PIL Image objects
        converted = []
        for img in images:
            if isinstance(img, bytes):
                img = Image.open(io.BytesIO(img))
            img = img.convert("RGB")
            converted.append(img)
        inputs = processor(converted, return_tensors="pt")
        inputs["labels"] = examples["label"]
        return inputs
    return _transform


def make_augmented_transform(processor):
    """
    Return a batched transform with data augmentation applied *before*
    the model's image processor.  Used for training only.

    Augmentations (all stochastic, applied per-image):
      - Random horizontal flip (50 %)
      - Random JPEG re-compression at quality 50-95 (30 %)
      - Random brightness / contrast jitter ±20 % (30 %)
    """
    import random
    from PIL import ImageEnhance

    def _augment_single(img):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # JPEG compression simulation — injects realistic artifacts
        if random.random() < 0.3:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=random.randint(50, 95))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        # Brightness / contrast jitter
        if random.random() < 0.3:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
        
        # NEW: Random rotation (small angles — real photos are rarely perfectly level)
        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, fillcolor=(128, 128, 128))
       
        # NEW: Random crop + resize (simulates different crops/shares of the same image)
        if random.random() < 0.3:
            w, h = img.size
            crop_ratio = random.uniform(0.8, 1.0)
            new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
            left = random.randint(0, w - new_w)
            top = random.randint(0, h - new_h)
            img = img.crop((left, top, left + new_w, top + new_h))
            img = img.resize((w, h), Image.BILINEAR)
        
        # NEW: Random Gaussian blur (simulates real camera blur/social media compression)
        if random.random() < 0.2:
            from PIL import ImageFilter
            radius = random.uniform(0.5, 1.5)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        
        # NEW: Color jitter (saturation + hue)
        if random.random() < 0.3:
            img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3))
        
        return img

    def _transform(examples):
        images = examples["image"]
        if not isinstance(images, list):
            images = [images]
        converted = []
        for img in images:
            if isinstance(img, bytes):
                img = Image.open(io.BytesIO(img))
            img = img.convert("RGB")
            img = _augment_single(img)
            converted.append(img)
        inputs = processor(converted, return_tensors="pt")
        inputs["labels"] = examples["label"]
        return inputs

    return _transform


def collate_fn(examples):
    """Custom collator – stacks pixel_values & labels into tensors."""
    pixel_values = []
    for ex in examples:
        pv = ex["pixel_values"]
        if isinstance(pv, list):
            pv = torch.tensor(pv)
        pixel_values.append(pv)
    return {
        "pixel_values": torch.stack(pixel_values),
        "labels": torch.tensor([ex["labels"] for ex in examples]),
    }


# ── Training ──────────────────────────────────────────────────────────────────

def run_training_stage(
    model,
    processor,
    train_ds,
    eval_ds,
    output_dir,
    epochs=3,
    batch_size=8,
    learning_rate=2e-5,
    stage_name="stage",
    fp16=False,
    replay_ds=None,
    replay_ratio=0.25,
    freeze_layers=None,
    augment=False,
    weight_decay=0.05,
    early_stopping_patience=2,
):
    """
    Fine-tune *model* on *train_ds* / *eval_ds* and save to *output_dir*.

    If *replay_ds* is provided, a random subset (sized as *replay_ratio* ×
    len(train_ds)) is drawn from it and concatenated with *train_ds* before
    training.  This implements **experience replay** to mitigate catastrophic
    forgetting when fine-tuning across sequential stages.

    If *freeze_layers* is set (e.g. 10), the first N ViT encoder layers plus
    the embedding layer are frozen so the model retains its general visual
    features and only the top layers + classifier head are updated.

    If *augment* is True, training images receive random flips, colour
    jitter, and JPEG re-compression to improve generalisation.

    Returns (trained_model, trainer).
    """
    print(f"\n{'='*60}")
    print(f"  🚀  {stage_name}")
    print(f"{'='*60}")

    # ── Experience replay: mix previous-stage data into current stage ──
    if replay_ds is not None:
        n_replay = max(1, int(len(train_ds) * replay_ratio))

        # Reset any transform set by a previous training stage so the
        # original column names (e.g. "label") are accessible for filtering.
        replay_ds.reset_format()

        # Balanced replay: 50/50 real/AI to prevent class bias
        real_replay = replay_ds.filter(lambda x: x["label"] == 0).shuffle(seed=42)
        ai_replay = replay_ds.filter(lambda x: x["label"] == 1).shuffle(seed=42)
        n_each = n_replay // 2
        replay_subset = concatenate_datasets([
            real_replay.select(range(min(n_each, len(real_replay)))),
            ai_replay.select(range(min(n_each, len(ai_replay)))),
        ]).shuffle(seed=42)

        # Align schemas — keep only columns common to both datasets
        keep = set(train_ds.column_names) & set(replay_subset.column_names)
        drop_primary = [c for c in train_ds.column_names if c not in keep]
        drop_replay  = [c for c in replay_subset.column_names if c not in keep]
        if drop_primary:
            train_ds = train_ds.remove_columns(drop_primary)
        if drop_replay:
            replay_subset = replay_subset.remove_columns(drop_replay)

        # Cast replay features to match train_ds features (e.g. binary → Image)
        # so that concatenate_datasets doesn't choke on type mismatches.
        replay_subset = replay_subset.cast(train_ds.features)

        train_ds = concatenate_datasets([train_ds, replay_subset]).shuffle(seed=42)
        print(f"  🔁  Experience replay: added {len(replay_subset)} samples "
              f"from previous stage ({replay_ratio:.0%} ratio) [balanced 50/50]")
        print(f"  📊  Combined training set: {len(train_ds)} samples")

    # ── Layer freezing: preserve general features, adapt only top layers ──
    if freeze_layers is not None:
        freeze_encoder_layers(model, num_layers_to_freeze=freeze_layers)

    # Apply transforms (with optional augmentation for training)
    if augment:
        train_ds.set_transform(make_augmented_transform(processor))
    else:
        train_ds.set_transform(make_transform(processor))
    eval_ds.set_transform(make_transform(processor))

    args = TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=fp16,
        push_to_hub=False,
        report_to="none",
        label_smoothing_factor=0.1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=weight_decay,
    )

    # ── Early stopping ──
    callbacks = []
    if early_stopping_patience is not None:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
        print(f"  ⏱️  Early stopping enabled (patience={early_stopping_patience})")

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collate_fn,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
        compute_metrics=compute_metrics,
        callbacks=callbacks or None,
    )

    results = trainer.train()
    trainer.save_model(output_dir)
    trainer.log_metrics("train", results.metrics)
    trainer.save_metrics("train", results.metrics)
    print(f"  ✅  Model saved → {output_dir}")
    return model, trainer


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    processor,
    test_ds,
    output_prefix,
    output_dir="outputs",
    batch_size=8,
    fp16=False,
    label_names=None,
):
    """
    Evaluate *model* on *test_ds* and save JSON metrics + confusion-matrix PNG.

    Parameters
    ----------
    output_prefix : str   e.g. 'genimage_stage1' — used for file naming.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if label_names is None:
        label_names = ["Real", "AI-Generated"]

    os.makedirs(output_dir, exist_ok=True)

    transform = make_transform(processor)
    test_ds.set_transform(transform)

    args = TrainingArguments(
        output_dir=os.path.join(output_dir, "tmp_eval"),
        per_device_eval_batch_size=batch_size,
        remove_unused_columns=False,
        fp16=fp16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collate_fn,
        processing_class=processor,
        compute_metrics=compute_metrics,
    )

    predictions = trainer.predict(test_ds)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    # ── metrics ──
    report = classification_report(
        labels, preds, target_names=label_names, output_dict=True
    )
    cm = confusion_matrix(labels, preds)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    json_path = os.path.join(output_dir, f"{output_prefix}_eval_results.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  📄  Metrics saved → {json_path}")

    # ── confusion matrix plot ──
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(label_names); ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix – {output_prefix}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    png_path = os.path.join(output_dir, f"{output_prefix}_confusion_matrix.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  🖼️   Confusion matrix saved → {png_path}")

    # ── print summary ──
    print(f"\n  {'─'*40}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  {'─'*40}\n")
    print(classification_report(labels, preds, target_names=label_names))
    return metrics
