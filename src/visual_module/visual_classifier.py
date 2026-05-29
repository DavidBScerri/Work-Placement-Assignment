import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np

class VisualClassifier:
    def __init__(
        self,
        model_name_or_path="dima806/ai_vs_human_generated_image_detection",
        delta_path=None,
    ):
        """
        Initializes the visual classifier.

        Args:
            model_name_or_path: HuggingFace model ID or local path to a full model.
            delta_path: Optional path to a .pt delta file produced by save_weight_delta().
                        When provided, the base model is loaded from model_name_or_path
                        and the stored weight differences are applied on top, avoiding
                        the need to store a full 327 MB model file in the repository.
        """
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name_or_path,
            attn_implementation="eager"
        ).to(self.device)

        if delta_path is not None:
            print(f"Applying weight delta from '{delta_path}'...")
            load_weight_delta(self.model, delta_path, device=self.device)
            print("Delta applied successfully.")

        self.model.eval()

    def predict(self, image):
        """
        Predicts whether an image is AI generated, Real, or Uncertain.

        Args:
            image: A PIL Image object.
            uncertainty_threshold: Confidence threshold below which the model is "Uncertain".

        Returns:
            A dictionary containing the prediction label and confidence score.
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
        max_prob, predicted_class_idx = torch.max(probs, dim=-1)
        max_prob = max_prob.item()
        predicted_class_idx = predicted_class_idx.item()
        
        # Get label from model config
        label = self.model.config.id2label[predicted_class_idx].lower()
        
        # Map label to readable format
        if "human" in label or "real" in label:
            mapped_label = "Real"
        else:
            mapped_label = "AI Generated"
            
        return {
            "prediction": mapped_label,
            "confidence": round(max_prob, 4),
            "raw_label": label,
            "all_scores": {self.model.config.id2label[i]: round(probs[0][i].item(), 4) for i in range(len(probs[0]))}
        }


# ---------------------------------------------------------------------------
# Weight-delta helpers
# ---------------------------------------------------------------------------

def save_weight_delta(
    fine_tuned_model,
    base_model_name="dima806/ai_vs_human_generated_image_detection",
    output_path="./fine_tuned_model_delta/weight_delta.pt",
    threshold: float = 1e-9,
):
    """
    Saves weight differences between the fine-tuned model and the base model
    using per-tensor int8 quantisation.

    Encoding:
        For each parameter tensor whose L∞ change exceeds `threshold`:
            scale  = max_abs_diff / 127.0
            stored = round(diff / scale).clamp(-127, 127)  [int8]
        Reconstruction (done in load_weight_delta):
            diff   ≈ stored.float() * scale

    Results for this ViT (86 M params, all layers unfrozen):
        Full model.safetensors : 327 MB  (❌ over GitHub 100 MB limit)
        This delta file        :  ~82 MB  (✅ under GitHub 100 MB limit)
        Max reconstruction err : < 0.00002  (negligible vs weight scale ~0.01–1.0)

    Args:
        fine_tuned_model: The trained model object (in memory after fine_tune_model()).
        base_model_name:  HuggingFace model ID of the base model used for training.
        output_path:      Where to write the .pt delta file.
        threshold:        Tensors whose L∞ change is below this are skipped (pure zeros).
    Returns:
        (output_path, size_mb)
    """
    print(f"Loading base model '{base_model_name}' to compute delta...")
    base_model = AutoModelForImageClassification.from_pretrained(base_model_name)

    # Move both state dicts to CPU upfront to avoid any device mismatch
    ft_state   = {k: v.float().cpu() for k, v in fine_tuned_model.state_dict().items()}
    base_state = {k: v.float().cpu() for k, v in base_model.state_dict().items()}
    del base_model  # free memory

    delta     = {}
    unchanged = []
    for key in ft_state:
        ft_param   = ft_state[key]
        base_param = base_state[key] if key in base_state else torch.zeros_like(ft_param)
        diff       = ft_param - base_param
        max_abs    = diff.abs().max().item()
        if max_abs < threshold:
            unchanged.append(key)
            continue
        # Per-tensor int8 quantisation — 4× smaller than float32, 2× smaller than float16
        scale = max_abs / 127.0
        quant = (diff / scale).round().clamp(-127, 127).to(torch.int8)
        delta[key] = {"q": quant, "s": scale}

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(
        {"base_model": base_model_name, "dtype": "int8", "delta": delta},
        output_path,
    )

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Delta saved → '{output_path}'")
    print(f"  Changed parameters : {len(delta)}")
    print(f"  Unchanged (skipped): {len(unchanged)}")
    print(f"  File size          : {size_mb:.2f} MB")
    return output_path, size_mb


def load_weight_delta(model, delta_path, device=None):
    """
    Applies a weight delta (produced by save_weight_delta) to an already-loaded
    base model, modifying it in-place.

    Supports both the int8-quantised format ({"q": int8_tensor, "s": scale})
    and the legacy float16 format (raw half-precision tensor) for backwards
    compatibility.

    Args:
        model:      The base model instance — weights are updated in-place.
        delta_path: Path to the .pt file created by save_weight_delta().
        device:     torch.device to map tensors onto (defaults to CPU).
    """
    checkpoint = torch.load(delta_path, map_location=device or "cpu", weights_only=False)
    delta      = checkpoint["delta"]
    fmt        = checkpoint.get("dtype", "float16")
    state      = model.state_dict()

    for key, payload in delta.items():
        # Backward compatibility for legacy sequential classifier head keys
        target_key = key
        if key == "classifier.1.weight":
            target_key = "classifier.weight"
        elif key == "classifier.1.bias":
            target_key = "classifier.bias"

        if target_key not in state:
            continue
        if fmt == "int8" and isinstance(payload, dict):
            # Dequantise: diff ≈ q * scale
            diff = payload["q"].float() * payload["s"]
        else:
            # Legacy float16 delta
            diff = payload.float()
        state[target_key] = (state[target_key].float() + diff).to(state[target_key].dtype)

    model.load_state_dict(state)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def fine_tune_model(
    model_name="dima806/ai_vs_human_generated_image_detection",
    dataset_name="julienlucas/midjourney-dalle-sd-nanobananapro-dataset",
    output_dir="./fine_tuned_model",
    epochs=3,
    batch_size=16,
    learning_rate=2e-5
):
    """
    Fine-tunes the image classification model on the provided dataset.
    """
    print(f"Loading dataset: {dataset_name}")
    if "GenImage" in dataset_name:
        from datasets import interleave_datasets
        print("Using streaming mode for GenImage to preserve local storage.")
        dataset = load_dataset(dataset_name, streaming=True)
        
        def map_label(example):
            path = example.get("image_path", "").lower()
            # dima806 model: 0 is human/real, 1 is AI-generated
            example["label"] = 0 if "/nature/" in path or "/real/" in path else 1
            return example
            
        mapped_ds = dataset['train'].map(map_label)
        
        real_stream = mapped_ds.filter(lambda x: x["label"] == 0)
        fake_stream = mapped_ds.filter(lambda x: x["label"] == 1)
        
        balanced_stream = interleave_datasets([real_stream, fake_stream])
        
        # Take 50,000 unique images for training
        train_ds = balanced_stream.take(50000)
        # Skip 50,000 and take 20,000 for testing
        test_ds = balanced_stream.skip(50000).take(20000)
        
        max_train_samples = 50000
    else:
        dataset = load_dataset(dataset_name)
        train_ds = dataset['train']
        test_ds = dataset['test']
        max_train_samples = len(train_ds)
    
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name, ignore_mismatched_sizes=True)
    
    # Make sure we have label mappings correctly depending on the dataset
    # You might need to map dataset labels to the model's labels if they differ.
    # Assuming dataset has 'image' and 'label' columns.
    
    def transforms(examples):
        # Support both batched lists or individual dicts
        images = examples["image"] if isinstance(examples["image"], list) else [examples["image"]]
        images = [img.convert("RGB") for img in images]
        
        inputs = processor(images, return_tensors="pt")
        inputs["labels"] = examples["label"]
        return inputs
        
    print("Applying transformations...")
    if "GenImage" in dataset_name:
        cols_to_remove = ["image", "image_path", "md5", "width", "height"]
        train_ds = train_ds.map(transforms, batched=True, batch_size=batch_size, remove_columns=cols_to_remove)
        test_ds = test_ds.map(transforms, batched=True, batch_size=batch_size, remove_columns=cols_to_remove)
    else:
        train_ds.set_transform(transforms)
        test_ds.set_transform(transforms)
    
    # Calculate max steps for streaming datasets
    gradient_accumulation_steps = 4
    steps_per_epoch = max_train_samples // (batch_size * gradient_accumulation_steps)
    max_steps = steps_per_epoch * epochs if "GenImage" in dataset_name else -1
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        eval_strategy="steps" if "GenImage" in dataset_name else "epoch",
        eval_steps=steps_per_epoch if "GenImage" in dataset_name else None,
        save_strategy="steps" if "GenImage" in dataset_name else "epoch",
        save_steps=steps_per_epoch if "GenImage" in dataset_name else None,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps, # To fit in VRAM effectively
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        max_steps=max_steps,
        warmup_steps=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )
    
    def collate_fn(examples):
        # Handle case where pixel_values might be lists depending on mapping
        pixel_values = []
        for example in examples:
            pv = example["pixel_values"]
            if isinstance(pv, list):
                pv = torch.tensor(pv)
            pixel_values.append(pv)
        pixel_values = torch.stack(pixel_values)
        labels = torch.tensor([example["labels"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=processor,
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    train_results = trainer.train()
    
    print("Saving model...")
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    
    return model, processor

def get_genimage_test_dataset():
    """
    Returns the streaming test dataset for GenImage, matching the split logic
    used during fine-tuning (skip 50000, take 20000).
    """
    from datasets import load_dataset, interleave_datasets
    dataset = load_dataset("nebula/GenImage-arrow", streaming=True)
    
    def map_label(example):
        path = example.get("image_path", "").lower()
        example["label"] = 0 if "/nature/" in path or "/real/" in path else 1
        return example
        
    mapped_ds = dataset['train'].map(map_label)
    real_stream = mapped_ds.filter(lambda x: x["label"] == 0)
    fake_stream = mapped_ds.filter(lambda x: x["label"] == 1)
    
    balanced_stream = interleave_datasets([real_stream, fake_stream])
    test_ds = balanced_stream.skip(50000).take(20000)
    return test_ds

