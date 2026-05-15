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
from datasets import load_dataset, Dataset, DatasetDict
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
):
    """
    Fine-tune *model* on *train_ds* / *eval_ds* and save to *output_dir*.
    Returns (trained_model, trainer).
    """
    print(f"\n{'='*60}")
    print(f"  🚀  {stage_name}")
    print(f"{'='*60}")

    # Apply transforms
    transform = make_transform(processor)
    train_ds.set_transform(transform)
    eval_ds.set_transform(transform)

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
        warmup_steps=50,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=fp16,
        push_to_hub=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collate_fn,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
        compute_metrics=compute_metrics,
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
