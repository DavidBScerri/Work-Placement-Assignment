import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np

class VisualClassifier:
    def __init__(self, model_name_or_path="dima806/ai_vs_human_generated_image_detection"):
        """
        Initializes the visual classifier with the given model.
        """
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        self.model = AutoModelForImageClassification.from_pretrained(model_name_or_path).to(self.device)
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
    dataset = load_dataset(dataset_name)
    
    # Extract train and test sets
    train_ds = dataset['train']
    test_ds = dataset['test']
    
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name, ignore_mismatched_sizes=True)
    
    # Make sure we have label mappings correctly depending on the dataset
    # You might need to map dataset labels to the model's labels if they differ.
    # Assuming dataset has 'image' and 'label' columns.
    
    def transforms(examples):
        inputs = processor([img.convert("RGB") for img in examples["image"]], return_tensors="pt")
        inputs["labels"] = examples["label"]
        return inputs
        
    print("Applying transformations...")
    train_ds.set_transform(transforms)
    test_ds.set_transform(transforms)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4, # To fit in VRAM effectively
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        warmup_steps=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        use_mps_device=True if torch.backends.mps.is_available() else False,
    )
    
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
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

