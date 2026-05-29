import os
import sys
import json
import re
import io
import tempfile
import socket
import webbrowser
import threading
import base64
import atexit
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from PIL import Image

# Add project root to sys.path so we can import from src.*
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import pipeline dependencies
from src.metadata_module import analyse_image, AnalysisResult
from src.integration_pipeline.fusion import (
    get_fusion_strategy,
    extract_visual_ai_probability,
    crop_face_region,
)
from src.visual_module.gradcam import generate_gradcam_overlay

# ---------------------------------------------------------------------------
# Global State for Server Tracking
# ---------------------------------------------------------------------------
_active_server = None
_server_thread = None


# ---------------------------------------------------------------------------
# Pipeline Request Handler
# ---------------------------------------------------------------------------
class PipelineRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress logging request noise to keep the notebook output clean
        pass

    def do_GET(self):
        # Serve the single-page HTML application
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            
            html_path = Path(__file__).parent / "index.html"
            if html_path.exists():
                self.wfile.write(html_path.read_bytes())
            else:
                self.wfile.write(b"<h1>index.html not found</h1>")
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self):
        # Handle pipeline execution requests
        if self.path == "/api/analyse":
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Content-Type must be multipart/form-data"}).encode("utf-8"))
                return

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            # Extract multipart boundary
            boundary_match = re.search(r'boundary=([^;]+)', content_type)
            if not boundary_match:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No boundary found in Content-Type"}).encode("utf-8"))
                return

            boundary = b"--" + boundary_match.group(1).encode("utf-8")
            parts = body.split(boundary)

            file_data = None
            params = {}

            # Manually parse fields (avoids deprecated cgi module in Python 3.13+)
            for part in parts:
                if not part.strip() or part == b"--\r\n" or part == b"--":
                    continue
                
                if b'\r\n\r\n' not in part:
                    continue
                
                header_section, value_section = part.split(b'\r\n\r\n', 1)
                
                # Clean up leading/trailing linebreaks
                if header_section.startswith(b'\r\n'):
                    header_section = header_section[2:]
                if value_section.endswith(b'\r\n'):
                    value_section = value_section[:-2]

                headers_str = header_section.decode('utf-8', errors='ignore')

                name_match = re.search(r'name="([^"]+)"', headers_str)
                if not name_match:
                    continue
                name = name_match.group(1)

                if 'filename="' in headers_str:
                    file_data = value_section
                else:
                    params[name] = value_section.decode('utf-8', errors='ignore')

            if file_data is None:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No file uploaded"}).encode("utf-8"))
                return

            try:
                # Retrieve models from server object
                visual_classifier = self.server.visual_classifier
                deepfake_classifier = self.server.deepfake_classifier

                # Run pipeline
                result = run_analysis_pipeline(file_data, params, visual_classifier, deepfake_classifier)
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))


# ---------------------------------------------------------------------------
# Core Analysis Wrapper
# ---------------------------------------------------------------------------
def run_analysis_pipeline(file_data, params, visual_classifier, deepfake_classifier):
    # Load as PIL Image
    pil_image = Image.open(io.BytesIO(file_data))
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Step 1: Metadata Module (needs path to run exiftool)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(file_data)
        tmp_path = tmp.name

    try:
        meta_result = analyse_image(tmp_path)
    finally:
        os.unlink(tmp_path)

    # Step 2: Visual Classifier (Whole Image)
    visual_result = visual_classifier.predict(pil_image)
    visual_ai_prob = extract_visual_ai_probability(visual_result)

    # Step 2b: GradCAM heatmap for whole image
    try:
        visual_gradcam_b64 = generate_gradcam_overlay(
            model=visual_classifier.model,
            processor=visual_classifier.processor,
            image=pil_image,
            device=visual_classifier.device,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"[GradCAM] Whole-image heatmap failed: {e}")
        visual_gradcam_b64 = None

    # Step 3: Face crop detection & classification (unconditional)
    face_res = deepfake_classifier.predict_face(pil_image)
    bbox = face_res.get("bbox")
    
    cropped_visual_result = None
    cropped_visual_ai_prob = None
    cropped_face_b64 = None
    cropped_gradcam_b64 = None
    
    if bbox is not None:
        face_padding = float(params.get("face_padding", 0.30))
        cropped_face = crop_face_region(pil_image, bbox, padding=face_padding)
        
        # Save cropped face to base64
        buffered = io.BytesIO()
        cropped_face.save(buffered, format="JPEG")
        cropped_face_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        cropped_visual_result = visual_classifier.predict(cropped_face)
        cropped_visual_ai_prob = extract_visual_ai_probability(cropped_visual_result)

        # GradCAM heatmap for cropped face
        try:
            cropped_gradcam_b64 = generate_gradcam_overlay(
                model=visual_classifier.model,
                processor=visual_classifier.processor,
                image=cropped_face,
                device=visual_classifier.device,
            )
        except Exception as e:
            traceback.print_exc()
            print(f"[GradCAM] Cropped-face heatmap failed: {e}")
            cropped_gradcam_b64 = None

    # Step 4: Decision Fusion
    strategy_name = params.get("fusion_strategy", "weighted_average")
    if strategy_name == "weighted_average":
        strategy = get_fusion_strategy(
            "weighted_average",
            w_meta=float(params.get("w_meta", 0.30)),
            w_visual=float(params.get("w_visual", 0.70)),
            decision_threshold=float(params.get("wa_threshold", 0.50)),
            meta_accuracy=float(params.get("meta_accuracy", 0.70)),
            visual_accuracy=float(params.get("visual_accuracy", 0.83)),
        )
    elif strategy_name == "conservative_threshold":
        strategy = get_fusion_strategy(
            "conservative_threshold",
            meta_threshold=float(params.get("ct_meta_thresh", 0.70)),
            visual_threshold=float(params.get("ct_visual_thresh", 0.65)),
        )
    elif strategy_name == "bayesian":
        strategy = get_fusion_strategy(
            "bayesian",
            prior=float(params.get("bayes_prior", 0.50)),
            decision_threshold=float(params.get("bayes_threshold", 0.50)),
        )
    else:
        raise ValueError(f"Unknown fusion strategy '{strategy_name}'")

    fusion_result = strategy.fuse(
        metadata_ai_prob=meta_result.ai_probability,
        visual_ai_prob=visual_ai_prob,
        cropped_visual_ai_prob=cropped_visual_ai_prob,
    )

    # Step 5: Conditional Deepfake Analysis
    deepfake_result_data = None
    if fusion_result.is_ai:
        deepfake_result = deepfake_classifier.predict(pil_image)
        da = deepfake_result.get("deepfake_analysis")
        
        deepfake_threshold = float(params.get("deepfake_threshold", 0.50))
        
        has_face = da.get("has_face", False) if da else False
        is_face_deepfake = False
        if has_face and cropped_visual_ai_prob is not None:
            if cropped_visual_ai_prob >= deepfake_threshold:
                is_face_deepfake = True

        has_place = da.get("has_place", False) if da else False
        is_place_deepfake = False
        if has_place:
            landmark_conf = da.get("landmark_analysis", {}).get("confidence", 0.0)
            if landmark_conf >= deepfake_threshold:
                is_place_deepfake = True

        if is_face_deepfake or is_place_deepfake:
            verdict = "Probable Deepfake (AI-generated image containing identifiable face/place)"
            verdict_type = "deepfake"
        else:
            verdict = "Probably AI-generated, but not necessarily a deepfake (no face or landmark detected)"
            verdict_type = "ai_generated"

        deepfake_result_data = {
            "has_face": has_face,
            "has_place": has_place,
            "face_analysis": da.get("face_analysis") if da else None,
            "scene_analysis": da.get("scene_analysis") if da else None,
            "landmark_analysis": da.get("landmark_analysis") if da else None,
        }
    else:
        verdict = f"Probably Real (confidence: {1 - fusion_result.ai_probability:.2%})"
        verdict_type = "real"

    # Map pydantic models to serializable dicts
    meta_features = meta_result.features
    meta_features_dict = {
        "has_make": meta_features.has_make,
        "has_model": meta_features.has_model,
        "has_lens_model": meta_features.has_lens_model,
        "has_makernote": meta_features.has_makernote,
        "has_gps": meta_features.has_gps,
        "has_c2pa": meta_features.has_c2pa,
        "has_ai_claim": meta_features.has_ai_claim,
        "has_camera_claim": meta_features.has_camera_claim,
        "has_edit_claim": meta_features.has_edit_claim,
        "suspicious_only_software_tags": meta_features.suspicious_only_software_tags,
        "suspicious_perfect_timestamp": meta_features.suspicious_perfect_timestamp,
    }

    return {
        "verdict": verdict,
        "verdict_type": verdict_type,
        "fusion": {
            "probability": fusion_result.ai_probability,
            "is_ai": fusion_result.is_ai,
            "strategy_name": fusion_result.formula_name,
            "explanation": fusion_result.explanation,
        },
        "metadata": {
            "probability": meta_result.ai_probability,
            "decision": meta_result.decision,
            "rationale": meta_result.rationale,
            "features": meta_features_dict,
        },
        "visual": {
            "probability": visual_ai_prob,
            "prediction": visual_result["prediction"],
            "confidence": visual_result["confidence"],
            "all_scores": visual_result["all_scores"],
            "gradcam_b64": visual_gradcam_b64,
        },
        "cropped_visual": {
            "probability": cropped_visual_ai_prob,
            "prediction": cropped_visual_result["prediction"] if cropped_visual_result else None,
            "confidence": cropped_visual_result["confidence"] if cropped_visual_result else None,
            "all_scores": cropped_visual_result["all_scores"] if cropped_visual_result else None,
            "cropped_face_b64": cropped_face_b64,
            "gradcam_b64": cropped_gradcam_b64 if cropped_visual_result else None,
        } if cropped_visual_result else None,
        "deepfake": deepfake_result_data,
    }


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def find_free_port():
    for port in range(5000, 6000):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except socket.error:
                continue
    raise RuntimeError("Could not find an available port in the 5000-6000 range.")


class PipelineHTTPServer(HTTPServer):
    def __init__(self, server_address, RequestHandlerClass, visual_classifier, deepfake_classifier):
        self.visual_classifier = visual_classifier
        self.deepfake_classifier = deepfake_classifier
        super().__init__(server_address, RequestHandlerClass)


def start_server_thread(visual_classifier, deepfake_classifier):
    global _active_server, _server_thread
    
    # Shut down existing server if active
    if _active_server is not None:
        print("Stopping existing web server...")
        _active_server.shutdown()
        _active_server.server_close()
        _active_server = None
        if _server_thread is not None:
            _server_thread.join()
            _server_thread = None

    port = find_free_port()
    server = PipelineHTTPServer(('127.0.0.1', port), PipelineRequestHandler, visual_classifier, deepfake_classifier)
    _active_server = server

    def serve():
        server.serve_forever()

    _server_thread = threading.Thread(target=serve, daemon=True)
    _server_thread.start()

    url = f"http://127.0.0.1:{port}"
    print(f"\n🚀 Seeing through Deepfakes web interface is live!")
    print(f"🔗 URL: {url}")
    print("Opening browser window automatically...")
    webbrowser.open(url)
    return url


@atexit.register
def stop_server():
    global _active_server
    if _active_server is not None:
        print("Shutting down active web server...")
        _active_server.shutdown()
        _active_server.server_close()
        _active_server = None


# ---------------------------------------------------------------------------
# Standalone Execution (for terminal testing)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting in Standalone mode. Initializing models...")
    from src.visual_module.visual_classifier import VisualClassifier
    from src.deepfake_module.deepfake_classifier import DeepfakeClassifier

    visual_delta_path = PROJECT_ROOT / "src" / "visual_module" / "fine_tuned_model_delta" / "run_02_ft_genimage_w_julienlucas_weight_delta.pt"
    deepfake_index_path = PROJECT_ROOT / "src" / "deepfake_module" / "models" / "landmarks_index.faiss"
    deepfake_meta_path  = PROJECT_ROOT / "src" / "deepfake_module" / "models" / "landmarks_metadata.json"

    print("Loading visual classifier...")
    visual_model = VisualClassifier(
        model_name_or_path="dima806/ai_vs_human_generated_image_detection",
        delta_path=str(visual_delta_path),
    )

    print("Loading deepfake classifier...")
    deepfake_model = DeepfakeClassifier(
        index_path=str(deepfake_index_path),
        metadata_path=str(deepfake_meta_path),
    )

    port = find_free_port()
    server = PipelineHTTPServer(('127.0.0.1', port), PipelineRequestHandler, visual_model, deepfake_model)
    print(f"\n🚀 Standalone server running at http://127.0.0.1:{port}")
    webbrowser.open(f"http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping standalone server...")
        server.shutdown()
        server.server_close()
