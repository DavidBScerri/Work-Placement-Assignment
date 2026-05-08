import os
import torch
import faiss
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

def build_landmark_index(dataset_name="zguo0525/google-landmarks-v2-mini",
                         model_name="facebook/dinov2-base",
                         output_dir="models"):
    """
    Downloads the Google Landmarks dataset, extracts DINOv2 embeddings,
    and builds a FAISS index for retrieval.
    """
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # 1. Load Dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    class_names = dataset.features["label"].names
    print(f"Dataset loaded. Total images: {len(dataset)}, Total classes: {len(class_names)}")

    # 2. Load Model
    print(f"Loading embedding model: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    labels = []

    # 3. Extract Embeddings
    print("Extracting embeddings (this may take a few minutes)...")
    for i, item in enumerate(tqdm(dataset)):
        image = item["image"]
        label = item["label"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # DINOv2 CLS token
            emb = outputs.last_hidden_state[:, 0, :]
            # Normalize for cosine similarity
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            embeddings.append(emb.cpu().numpy())
            labels.append(int(label))

    embeddings = np.vstack(embeddings).astype('float32')

    # 4. Build FAISS Index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    # IndexFlatIP with normalized vectors = Cosine Similarity
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # 5. Save Results
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "landmarks_index.faiss")
    metadata_path = os.path.join(output_dir, "landmarks_metadata.json")

    print(f"Saving index to {index_path}...")
    faiss.write_index(index, index_path)

    print(f"Saving metadata to {metadata_path}...")
    metadata = {
        "labels": labels,
        "class_names": class_names,
        "model_name": model_name
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

    print("Index initialization complete!")

if __name__ == "__main__":
    # Ensure we are in the right directory or provide absolute paths
    # For this task, we assume the script is run from src/deepfake_module
    build_landmark_index()
