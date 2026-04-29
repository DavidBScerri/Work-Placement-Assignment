"""
Imperceptible Watermark Module.

This module performs scheme-specific watermark evidence detection. 
It is important to note that this is not a universal watermark extraction tool.
Invisible watermark detection is scheme-specific and a valid result normally requires 
a known decoder, key, expected payload format, or statistical validity check.

Absence of detection must be recorded as absence of detectable evidence, 
not proof that the image is unwatermarked. Negative results mean only that 
no implemented decoder found valid evidence.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

from pydantic import BaseModel

# Try to import required packages, handle gracefully if not installed
try:
    import cv2
    import numpy as np
    from PIL import Image
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from imwatermark import WatermarkDecoder
    IMWATERMARK_AVAILABLE = True
except ImportError:
    IMWATERMARK_AVAILABLE = False

try:
    import pywt
    from scipy.fftpack import dct
    PYWT_SCIPY_AVAILABLE = True
except ImportError:
    PYWT_SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- Output Models ---

class WatermarkDetectionResult(BaseModel):
    scheme_id: str
    detector_name: str
    attempted: bool
    detected: bool
    confidence: Optional[float] = None
    decoded_payload: Optional[str] = None
    payload_valid: bool
    status: str
    notes: str
    error: Optional[str] = None

class WatermarkReport(BaseModel):
    image_path: str
    image_exists: bool
    image_loadable: bool
    implemented_schemes: List[str]
    results: List[WatermarkDetectionResult]
    overall_detectable_watermark_evidence: bool
    summary: str


class WatermarkDetector(Protocol):
    scheme_id: str
    detector_name: str

    def detect(self, image_path: Path, expected_payload: Optional[str] = None, **kwargs) -> WatermarkDetectionResult:
        ...

# --- Detectors ---

class DWTDCTDetector:
    scheme_id = "dwt_dct"
    detector_name = "invisible-watermark (DWT-DCT bytes)"

    def detect(self, image_path: Path, expected_payload: Optional[str] = None, payload_bit_lengths: Optional[List[int]] = None) -> WatermarkDetectionResult:
        if not IMWATERMARK_AVAILABLE or not OPENCV_AVAILABLE:
            return WatermarkDetectionResult(
                scheme_id=self.scheme_id,
                detector_name=self.detector_name,
                attempted=False,
                detected=False,
                payload_valid=False,
                status="unsupported_missing_dependencies",
                notes="invisible-watermark or opencv-python not installed.",
            )

        if payload_bit_lengths is None:
            payload_bit_lengths = [32, 64, 128, 256]

        try:
            bgr_img = cv2.imread(str(image_path))
            if bgr_img is None:
                raise ValueError("OpenCV failed to read the image.")
            
            for bit_length in payload_bit_lengths:
                decoder = WatermarkDecoder('bytes', bit_length)
                decoded_bytes = decoder.decode(bgr_img, 'dwtDct')
                
                if decoded_bytes:
                    try:
                        decoded_str = decoded_bytes.decode('utf-8', errors='ignore').strip('\x00')
                    except Exception:
                        decoded_str = str(decoded_bytes)
                    
                    if expected_payload is not None:
                        if expected_payload in decoded_str or decoded_str == expected_payload:
                            return WatermarkDetectionResult(
                                scheme_id=self.scheme_id,
                                detector_name=self.detector_name,
                                attempted=True,
                                detected=True,
                                confidence=1.0,
                                decoded_payload=decoded_str,
                                payload_valid=True,
                                status="expected_payload_matched",
                                notes=f"Successfully decoded and matched expected payload at {bit_length} bits."
                            )
                        else:
                            continue
                    else:
                        return WatermarkDetectionResult(
                            scheme_id=self.scheme_id,
                            detector_name=self.detector_name,
                            attempted=True,
                            detected=False,
                            decoded_payload=decoded_str,
                            payload_valid=False,
                            status="empty_or_unreadable_payload",
                            notes=f"Decoded payload at {bit_length} bits, but no expected payload provided to verify validity."
                        )

            return WatermarkDetectionResult(
                scheme_id=self.scheme_id,
                detector_name=self.detector_name,
                attempted=True,
                detected=False,
                confidence=0.0,
                payload_valid=False,
                status="no_detectable_evidence",
                notes="Attempted all specified bit lengths. No valid payload found or decoded."
            )

        except Exception as e:
            return WatermarkDetectionResult(
                scheme_id=self.scheme_id,
                detector_name=self.detector_name,
                attempted=True,
                detected=False,
                confidence=0.0,
                payload_valid=False,
                status="decoder_failed",
                notes="An exception occurred during decoding.",
                error=str(e)
            )


class ImageDWTDCTDetector:
    scheme_id = "image_dwt_dct"
    detector_name = "Image Watermark (DWT-DCT array)"

    def detect(self, image_path: Path, expected_payload: Optional[str] = None, **kwargs) -> WatermarkDetectionResult:
        if not PYWT_SCIPY_AVAILABLE or not OPENCV_AVAILABLE:
            return WatermarkDetectionResult(
                scheme_id=self.scheme_id,
                detector_name=self.detector_name,
                attempted=False,
                detected=False,
                payload_valid=False,
                status="unsupported_missing_dependencies",
                notes="pywt, scipy, or opencv-python not installed.",
            )

        try:
            # Load and resize image to 2048x2048, convert to grayscale as per the repo
            try:
                # Use PIL's built-in Resampling enum if available, fallback to ANTIALIAS
                resample_filter = getattr(Image, 'Resampling', Image).LANCZOS
                img = Image.open(image_path).resize((2048, 2048), resample_filter)
                img = img.convert('L')
                image_array = np.array(img.getdata(), dtype=float).reshape((2048, 2048))
            except Exception as e:
                raise ValueError(f"Failed to load and resize image: {e}")

            model = 'haar'
            level = 1
            
            # Process coefficients
            coeffs = pywt.wavedec2(data=image_array, wavelet=model, level=level)
            coeffs_H = list(coeffs)
            
            # Apply DCT on the LL band (coeffs_H[0])
            size = coeffs_H[0].shape[0]
            all_subdct = np.empty((size, size))
            for i in range(0, size, 8):
                for j in range(0, size, 8):
                    subpixels = coeffs_H[0][i:i+8, j:j+8]
                    subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
                    all_subdct[i:i+8, j:j+8] = subdct
                    
            # Extract the [5][5] coefficient from each 8x8 block to reconstruct the watermark
            subwatermarks = []
            for x in range(0, all_subdct.shape[0], 8):
                for y in range(0, all_subdct.shape[1], 8):
                    coeff_slice = all_subdct[x:x+8, y:y+8]
                    subwatermarks.append(coeff_slice[5][5])
            
            # The watermark size embedded by this algorithm is 128x128 
            # (since 2048 / 2 (DWT) = 1024; 1024 / 8 (block size) = 128)
            watermark_array = np.array(subwatermarks).reshape(128, 128)
            watermark_array_uint8 = np.uint8(watermark_array.clip(0, 255))
            
            # Compute variance to check if the extracted matrix is just uniform noise or an actual image
            var = np.var(watermark_array_uint8)
            
            if var > 10.0:  # simplistic threshold for non-uniform image
                return WatermarkDetectionResult(
                    scheme_id=self.scheme_id,
                    detector_name=self.detector_name,
                    attempted=True,
                    detected=True,
                    confidence=None,
                    decoded_payload="[Extracted 128x128 image array]",
                    payload_valid=True,
                    status="image_watermark_extracted",
                    notes=f"Successfully extracted an image array with variance {var:.2f}."
                )
            else:
                return WatermarkDetectionResult(
                    scheme_id=self.scheme_id,
                    detector_name=self.detector_name,
                    attempted=True,
                    detected=False,
                    confidence=0.0,
                    decoded_payload=None,
                    payload_valid=False,
                    status="no_detectable_evidence",
                    notes=f"Extracted array has low variance ({var:.2f}), likely no watermark."
                )

        except Exception as e:
            return WatermarkDetectionResult(
                scheme_id=self.scheme_id,
                detector_name=self.detector_name,
                attempted=True,
                detected=False,
                confidence=0.0,
                payload_valid=False,
                status="decoder_failed",
                notes="An exception occurred during decoding.",
                error=str(e)
            )


# --- Main analysis functions ---

def analyse_watermarks(
    image_path: Union[str, Path],
    expected_payloads: Optional[Dict[str, str]] = None,
    payload_bit_lengths: Optional[List[int]] = None
) -> WatermarkReport:
    """
    Analyses an image for scheme-specific watermark evidence.
    """
    path = Path(image_path)
    exists = path.exists()
    loadable = False
    
    if exists:
        try:
            from PIL import Image
            with Image.open(path) as img:
                img.verify()
            loadable = True
        except ImportError:
            # If PIL is missing, assume it might be loadable so detectors can report the missing dependency
            loadable = True
        except Exception:
            loadable = False

    detectors = [DWTDCTDetector(), ImageDWTDCTDetector()]
    implemented_schemes = [d.scheme_id for d in detectors]
    results = []

    overall_detectable_evidence = False

    if exists and loadable:
        for detector in detectors:
            expected_payload = None
            if expected_payloads and detector.scheme_id in expected_payloads:
                expected_payload = expected_payloads[detector.scheme_id]
                
            result = detector.detect(path, expected_payload=expected_payload, payload_bit_lengths=payload_bit_lengths)
            results.append(result)
            if result.detected:
                overall_detectable_evidence = True
    else:
        for detector in detectors:
            results.append(WatermarkDetectionResult(
                scheme_id=detector.scheme_id,
                detector_name=detector.detector_name,
                attempted=False,
                detected=False,
                payload_valid=False,
                status="skipped_image_unreadable",
                notes="Image does not exist or is not loadable."
            ))

    if overall_detectable_evidence:
        summary = "Detectable watermark evidence found by at least one implemented scheme."
    else:
        summary = "No known implemented scheme detected a valid watermark. Note: this does not prove the image is unwatermarked."

    return WatermarkReport(
        image_path=str(path),
        image_exists=exists,
        image_loadable=loadable,
        implemented_schemes=implemented_schemes,
        results=results,
        overall_detectable_watermark_evidence=overall_detectable_evidence,
        summary=summary
    )


def analyse_folder(folder_path: str, extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")) -> List[Dict[str, Any]]:
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return []

    results = []
    for file in sorted(folder.iterdir()):
        if file.suffix.lower() in extensions:
            try:
                report = analyse_watermarks(str(file))
                results.append(report.model_dump())
            except Exception as exc:
                results.append({"file": str(file), "error": str(exc)})
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python watermark.py <image_path>")
        sys.exit(1)

    try:
        res = analyse_watermarks(sys.argv[1])
        print(json.dumps(res.model_dump(), indent=2))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
