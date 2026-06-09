"""
Helper utilities for the two-stage fine-tuning notebook.
Stage 1: fine-tune on GenImage, Stage 2: continue on julienlucas.
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


def freeze_encoder_layers(model, num_layers_to_freeze=10):
    """
    Freeze the first N encoder layers of a ViT model, preserving general
    visual features while allowing the top layers and classifier head to adapt.

    Args:
        model: ViT model instance.
        num_layers_to_freeze: Number of layers to freeze from the bottom.
    """
    for param in model.vit.embeddings.parameters():
        param.requires_grad = False

    total_layers = len(model.vit.encoder.layer)
    n = min(num_layers_to_freeze, total_layers)
    for i in range(n):
        for param in model.vit.encoder.layer[i].parameters():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Froze {n}/{total_layers} encoder layers + embeddings")
    print(f"  Trainable: {trainable:,} / {total:,} params "
          f"({100 * trainable / total:.1f}%)")


BASE_MODEL_ID = "dima806/ai_vs_real_image_detection"


def load_model_from(source="base", device=None):
    """
    Load a model + processor from any source, ready for training or evaluation.

    Args:
        source: "base" for the original HuggingFace model, a local directory
                path, or a HuggingFace model ID.
        device: torch.device (auto-detected if None).

    Returns:
        (model, processor)
    """
    if device is None:
        device = get_device()

    model_id = BASE_MODEL_ID if source == "base" else source

    print(f"Loading model from: {model_id}")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(
        model_id, ignore_mismatched_sizes=True
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded - {n_params:,} parameters (device: {device})")
    return model, processor


def make_transform(processor):
    """
    Return a batched transform function compatible with HF Trainer.
    Converts images to RGB and applies the model's image processor.
    """
    def _transform(examples):
        images = examples["image"]
        if not isinstance(images, list):
            images = [images]
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
    Return a batched transform with data augmentation applied before
    the model's image processor. Used for training only.

    Augmentations: random flip, JPEG re-compression, brightness/contrast
    jitter, rotation, crop+resize, Gaussian blur, colour jitter.
    """
    import random
    from PIL import ImageEnhance

    def _augment_single(img):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.3:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=random.randint(50, 95))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        if random.random() < 0.3:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, fillcolor=(128, 128, 128))
        if random.random() < 0.3:
            w, h = img.size
            crop_ratio = random.uniform(0.8, 1.0)
            new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
            left = random.randint(0, w - new_w)
            top = random.randint(0, h - new_h)
            img = img.crop((left, top, left + new_w, top + new_h))
            img = img.resize((w, h), Image.BILINEAR)
        if random.random() < 0.2:
            from PIL import ImageFilter
            radius = random.uniform(0.5, 1.5)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
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
    """Custom collator - stacks pixel_values & labels into tensors."""
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
    Fine-tune model on train_ds/eval_ds and save to output_dir.

    Args:
        replay_ds:    If provided, a balanced subset is concatenated with
                      train_ds for experience replay.
        replay_ratio: Size of replay subset relative to train_ds.
        freeze_layers: If set, freeze the first N ViT encoder layers.
        augment:      If True, apply data augmentation during training.
        early_stopping_patience: Patience for early stopping (None to disable).

    Returns:
        (trained_model, trainer)
    """
    print(f"\n{'='*60}")
    print(f"  {stage_name}")
    print(f"{'='*60}")

    if replay_ds is not None:
        n_replay = max(1, int(len(train_ds) * replay_ratio))
        replay_ds.reset_format()

        real_replay = replay_ds.filter(lambda x: x["label"] == 0).shuffle(seed=42)
        ai_replay = replay_ds.filter(lambda x: x["label"] == 1).shuffle(seed=42)
        n_each = n_replay // 2
        replay_subset = concatenate_datasets([
            real_replay.select(range(min(n_each, len(real_replay)))),
            ai_replay.select(range(min(n_each, len(ai_replay)))),
        ]).shuffle(seed=42)

        keep = set(train_ds.column_names) & set(replay_subset.column_names)
        drop_primary = [c for c in train_ds.column_names if c not in keep]
        drop_replay  = [c for c in replay_subset.column_names if c not in keep]
        if drop_primary:
            train_ds = train_ds.remove_columns(drop_primary)
        if drop_replay:
            replay_subset = replay_subset.remove_columns(drop_replay)

        replay_subset = replay_subset.cast(train_ds.features)

        train_ds = concatenate_datasets([train_ds, replay_subset]).shuffle(seed=42)
        print(f"  Experience replay: added {len(replay_subset)} samples "
              f"from previous stage ({replay_ratio:.0%} ratio) [balanced 50/50]")
        print(f"  Combined training set: {len(train_ds)} samples")

    if freeze_layers is not None:
        freeze_encoder_layers(model, num_layers_to_freeze=freeze_layers)

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

    callbacks = []
    if early_stopping_patience is not None:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
        print(f"  Early stopping enabled (patience={early_stopping_patience})")

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
    print(f"  Model saved to {output_dir}")
    return model, trainer


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
    Evaluate model on test_ds and save JSON metrics + confusion matrix PNG.

    Args:
        output_prefix: String used for file naming (e.g. 'genimage_stage1').
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
    print(f"  Metrics saved to {json_path}")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(label_names); ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {output_prefix}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    png_path = os.path.join(output_dir, f"{output_prefix}_confusion_matrix.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved to {png_path}")

    print(f"\n  {'─'*40}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  {'─'*40}\n")
    print(classification_report(labels, preds, target_names=label_names))
    return metrics
