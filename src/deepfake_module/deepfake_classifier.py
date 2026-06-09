import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from datasets import load_dataset
import birder
import numpy as np
import faiss
import json
from PIL import Image
from facenet_pytorch import MTCNN


class DeepfakeClassifier:
    def __init__(self,
                 scene_model_name="birder-project/rope_vit_reg4_b14_capi-places365",
                 landmark_model_name="facebook/dinov2-base",
                 index_path="models/landmarks_index.faiss",
                 metadata_path="models/landmarks_metadata.json"):
        """
        Initializes the Deepfake Classifier with sub-models for face detection,
        scene classification, and landmark retrieval.

        Args:
            scene_model_name:     Birder model name for scene/Places365 classification.
            landmark_model_name:  HuggingFace model ID for landmark embeddings (DINOv2).
            index_path:           Path to the FAISS index file.
            metadata_path:        Path to the landmark metadata JSON file.
        """
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        print("Loading Face Detection model: MTCNN")
        self.mtcnn = MTCNN(keep_all=True, device='cpu')

        print(f"Loading Scene model: {scene_model_name}")
        try:
            (self.scene_model, self.scene_info) = birder.load_pretrained_model(scene_model_name, inference=True)
            self.scene_model.to(self.device)
            self.scene_model.eval()
            size = birder.get_size_from_signature(self.scene_info.signature)
            self.scene_transform = birder.classification_transform(size, self.scene_info.rgb_stats)
        except Exception as e:
            print(f"Warning: Could not load {scene_model_name} via birder: {e}. Falling back to google/vit-base-patch16-224.")
            self.scene_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
            self.scene_model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224").to(self.device)
            self.scene_model.eval()
            self.scene_info = None

        print(f"Loading Landmark Retrieval model: {landmark_model_name}")
        self.landmark_index = LandmarkIndex(
            model_name=landmark_model_name,
            index_path=index_path,
            metadata_path=metadata_path,
            device=self.device
        )

    def predict_face(self, image):
        """
        Detects whether a face is present in the image.

        Args:
            image: A PIL Image object.

        Returns:
            dict with keys ``label``, ``confidence``, and ``bbox``.
        """
        boxes, probs = self.mtcnn.detect(image)
        if boxes is None or len(boxes) == 0:
            return {"label": "No Face", "confidence": 0.0, "bbox": None}
        
        idx = int(np.argmax(probs))
        face_certainty = float(probs[idx])
        
        if face_certainty < 0.90:
            return {"label": "No Face", "confidence": round(face_certainty, 4), "bbox": None}

        bbox = [float(x) for x in boxes[idx]]
        return {"label": "Face Detected", "confidence": round(face_certainty, 4), "bbox": bbox}

    def predict_scene(self, image):
        """
        Classifies the generic scene in the image (Places365).

        Args:
            image: A PIL Image object.

        Returns:
            dict with keys ``label`` and ``confidence``.
        """
        if self.scene_info:
            with torch.no_grad():
                input_tensor = self.scene_transform(image).unsqueeze(0).to(self.device)
                outputs = self.scene_model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=-1)

            max_prob, idx = torch.max(probs, dim=-1)
            label = self.scene_info.labels[idx.item()]
        else:
            inputs = self.scene_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.scene_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            max_prob, idx = torch.max(probs, dim=-1)
            label = self.scene_model.config.id2label[idx.item()]

        return {"label": label, "confidence": round(max_prob.item(), 4)}

    def predict_landmark(self, image, top_k=10, similarity_threshold=0.5):
        """
        Identifies landmarks using DINOv2 embeddings and FAISS retrieval.

        Args:
            image:                A PIL Image object.
            top_k:                Number of nearest neighbors to retrieve.
            similarity_threshold: Minimum similarity score to consider a match.

        Returns:
            dict with keys ``label``, ``confidence``, and optionally ``top_matches``.
        """
        return self.landmark_index.search(image, top_k=top_k, similarity_threshold=similarity_threshold)

    def predict(self, image, visual_classifier=None, threshold=0.5):
        """
        Full pipeline: optionally gate on visual classifier confidence,
        then run all deepfake sub-models.

        Args:
            image:             A PIL Image object.
            visual_classifier: Optional VisualClassifier instance for gating.
            threshold:         Confidence threshold for triggering deepfake analysis.

        Returns:
            dict with keys ``visual_classification`` and ``deepfake_analysis``.
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')

        results = {
            "visual_classification": None,
            "deepfake_analysis": None
        }

        ai_score = threshold
        if visual_classifier:
            vis_res = visual_classifier.predict(image)
            results["visual_classification"] = vis_res
            ai_score = vis_res["confidence"] if vis_res["prediction"] == "AI Generated" else (1 - vis_res["confidence"])

        if ai_score >= threshold:
            face_res = self.predict_face(image)
            scene_res = self.predict_scene(image)
            landmark_res = self.predict_landmark(image)

            has_face = face_res["confidence"] >= 0.90
            has_landmark = landmark_res["label"] not in ["Unknown", "None", "N/A"] and landmark_res.get("confidence", 0.0) >= 0.50

            results["deepfake_analysis"] = {
                "has_face": has_face,
                "has_place": has_landmark,
                "face_analysis": face_res,
                "scene_analysis": scene_res,
                "landmark_analysis": landmark_res
            }

        return results


class LandmarkIndex:
    def __init__(self,
                 model_name="facebook/dinov2-base",
                 index_path="models/landmarks_index.faiss",
                 metadata_path="models/landmarks_metadata.json",
                 device=None):
        from transformers import AutoModel
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = None

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.load()
        else:
            print(f"Warning: Landmark index not found at {index_path}. "
                  "Please run the initialization script to build the FAISS index.")

    def load(self):
        print(f"Loading FAISS index from {self.index_path}...")
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        print("Landmark index loaded successfully.")

    def search(self, image, top_k=10, similarity_threshold=0.5):
        """Searches the FAISS index for the closest landmarks."""
        if self.index is None:
            return {"label": "N/A", "confidence": 0.0, "message": "Index not loaded."}

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            embedding_np = embedding.cpu().numpy().astype('float32')

        distances, indices = self.index.search(embedding_np, top_k)

        hits = {}
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            label_idx = self.metadata["labels"][idx]
            label_name = self.metadata["class_names"][label_idx]
            if label_name not in hits:
                hits[label_name] = []
            hits[label_name].append(float(dist))

        if not hits:
            return {"label": "None", "confidence": 0.0}

        best_label = None
        max_avg_sim = -1.0
        for label, sims in hits.items():
            avg_sim = sum(sims) / len(sims)
            if avg_sim > max_avg_sim:
                max_avg_sim = avg_sim
                best_label = label

        if max_avg_sim < similarity_threshold:
            return {"label": "Unknown", "confidence": round(max_avg_sim, 4)}

        return {
            "label": best_label,
            "confidence": round(max_avg_sim, 4),
            "matches_count": len(hits[best_label]),
            "all_matches": hits
        }
