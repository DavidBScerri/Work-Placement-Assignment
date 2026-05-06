import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoModel
import birder
from birder.inference.classification import infer_image
from PIL import Image
import numpy as np

class DeepfakeClassifier:
    def __init__(self, 
                 face_model_name="prithivMLmods/deepfake-detector-model-v1",
                 scene_model_name="birder-project/rope_vit_reg4_b14_capi-places365",
                 landmark_model_path=None):
        """
        Initializes the Deepfake Classifier with multiple sub-models.
        """
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Face Forensics Model
        print(f"Loading Face Forensics model: {face_model_name}")
        try:
            self.face_processor = AutoImageProcessor.from_pretrained(face_model_name)
        except Exception:
            print(f"Warning: Could not load specific processor for {face_model_name}. Falling back to default ViT processor.")
            self.face_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        
        self.face_model = AutoModelForImageClassification.from_pretrained(face_model_name).to(self.device)
        self.face_model.eval()
        
        # 2. Scene Classification Model (Places365)
        print(f"Loading Scene model: {scene_model_name}")
        try:
            (self.scene_model, self.scene_info) = birder.load_pretrained_model(scene_model_name, inference=True)
            self.scene_model.to(self.device)
            self.scene_model.eval()
            size = birder.get_size_from_signature(self.scene_info.signature)
            self.scene_transform = birder.classification_transform(size, self.scene_info.rgb_stats)
        except Exception as e:
            print(f"Warning: Could not load {scene_model_name} via birder: {e}. Falling back to default ViT.")
            self.scene_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
            self.scene_model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224").to(self.device)
            self.scene_model.eval()
            self.scene_info = None
        
        # 3. Landmark Detection Model (DINOv2)
        self.landmark_model_path = landmark_model_path
        if landmark_model_path:
            print(f"Loading fine-tuned Landmark model from: {landmark_model_path}")
            self.landmark_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self.landmark_model = AutoModelForImageClassification.from_pretrained(landmark_model_path).to(self.device)
            self.landmark_model.eval()
        else:
            print("Landmark model not provided. Initialize with facebook/dinov2-base if you plan to train/fine-tune.")
            self.landmark_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            # We don't load the classifier head yet if it's not fine-tuned
            self.landmark_model = None

    def predict_face(self, image):
        """
        Detects if the face in the image is manipulated.
        """
        inputs = self.face_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.face_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Assuming binary classification: 0: Real, 1: Fake (Check model config to be sure)
        # HrutikAdsare/deepfake-detector-faceforensics labels:
        # We'll use the id2label from the model
        max_prob, idx = torch.max(probs, dim=-1)
        label = self.face_model.config.id2label[idx.item()]
        return {"label": label, "confidence": max_prob.item(), "probs": probs[0].tolist()}

    def predict_scene(self, image):
        """
        Classifies the generic scene.
        """
        if self.scene_info:
            # Use birder inference
            with torch.no_grad():
                # Convert PIL image to tensor via transform
                input_tensor = self.scene_transform(image).unsqueeze(0).to(self.device)
                outputs = self.scene_model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=-1)
                
            max_prob, idx = torch.max(probs, dim=-1)
            label = self.scene_info.labels[idx.item()]
            return {"label": label, "confidence": max_prob.item()}
        else:
            # Fallback to standard transformers
            inputs = self.scene_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.scene_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            max_prob, idx = torch.max(probs, dim=-1)
            label = self.scene_model.config.id2label[idx.item()]
            return {"label": label, "confidence": max_prob.item()}

    def predict_landmark(self, image):
        """
        Detects specific landmarks using fine-tuned DINOv2.
        """
        if not self.landmark_model:
            return {"label": "N/A", "confidence": 0.0, "message": "Landmark model not loaded."}
            
        inputs = self.landmark_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.landmark_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        max_prob, idx = torch.max(probs, dim=-1)
        label = self.landmark_model.config.id2label[idx.item()]
        return {"label": label, "confidence": max_prob.item()}

    def predict(self, image, visual_classifier=None, metadata_analyzer=None, threshold=0.5):
        """
        Integrated flow as requested by the user:
        Input image -> Metadata extraction -> Visual classifier -> AI generated confidence -> Deepfake classifier
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        results = {
            "metadata": None,
            "visual_classification": None,
            "deepfake_analysis": None,
            "final_decision": "Inconclusive"
        }

        # 1. Extract Metadata
        if metadata_analyzer:
            # metadata_analyzer is expected to have an analyse_image method (or similar)
            # Since metadata.py uses file paths, we might need the path or a temporary file.
            # For now, we'll assume it's passed or handled externally if only PIL Image is available.
            pass

        # 2. Visual Classifier
        ai_score = 0
        if visual_classifier:
            vis_res = visual_classifier.predict(image)
            results["visual_classification"] = vis_res
            ai_score = vis_res["confidence"] if vis_res["prediction"] == "AI Generated" else (1 - vis_res["confidence"])
        
        # 3. IF above threshold: Deepfake Classifier
        if ai_score >= threshold:
            face_res = self.predict_face(image)
            scene_res = self.predict_scene(image)
            landmark_res = self.predict_landmark(image)
            
            # Combine scores
            deepfake_score = face_res["confidence"] if face_res["label"].lower() == "fake" else (1 - face_res["confidence"])
            
            results["deepfake_analysis"] = {
                "deepfake_confidence": round(deepfake_score, 4),
                "face_analysis": face_res,
                "scene_analysis": scene_res,
                "landmark_analysis": landmark_res
            }
            results["final_decision"] = "Potential Deepfake" if deepfake_score > 0.5 else "Likely AI Generated (Non-Deepfake)"
        else:
            results["final_decision"] = "Likely Real / Low AI Confidence"

        return results
