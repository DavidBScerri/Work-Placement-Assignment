"""
GradCAM for ViT — generates attention-gradient heatmaps for HuggingFace
ViT image classification models.

Uses CLS-token attention weights from the last self-attention layer,
weighted by per-patch gradient magnitude for class specificity.
Falls back to pure attention if gradient computation fails.
"""

from __future__ import annotations

import base64
import io
import math
from typing import Optional

import numpy as np
import torch
from PIL import Image


def compute_gradcam(
    model,
    processor,
    image: Image.Image,
    device: torch.device,
    target_class: Optional[int] = None,
) -> np.ndarray:
    """
    Computes an attention-gradient heatmap for a given image and ViT model.

    Args:
        model:        HuggingFace ViTForImageClassification model.
        processor:    The corresponding AutoImageProcessor.
        image:        PIL Image (RGB).
        device:       torch.device to run on.
        target_class: Class index for gradients (None = predicted class).

    Returns:
        cam: numpy array of shape (grid_h, grid_w) with values in [0, 1].
    """
    model.eval()

    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    cam = None

    try:
        with torch.enable_grad():
            outputs = model(**inputs, output_attentions=True)
            logits = outputs.logits

            if target_class is None:
                target_class = logits.argmax(dim=-1).item()

            attn_tuple = outputs.attentions
            if attn_tuple is not None and len(attn_tuple) > 0:
                last_attn = attn_tuple[-1]
                last_attn.retain_grad()

                model.zero_grad()
                target_logit = logits[0, target_class]
                target_logit.backward()

                if last_attn.grad is not None:
                    attn = last_attn[0, :, 0, 1:]  # (heads, num_patches)
                    grad = last_attn.grad[0, :, 0, 1:]

                    cam = (grad.clamp(min=0) * attn).mean(dim=0)
                    cam = torch.relu(cam)
                    cam = cam.detach().cpu().numpy()
                else:
                    print("[GradCAM] Warning: no attention gradients available, using attention weights only")
                    attn = last_attn[0, :, 0, 1:].detach()
                    cam = attn.mean(dim=0).cpu().numpy()
            else:
                print("[GradCAM] Warning: no attention data available. Check if model was loaded with attn_implementation='eager'")
    except Exception as e:
        print(f"[GradCAM] Error during computation: {e}")
        import traceback
        traceback.print_exc()

    if cam is None:
        return np.zeros((14, 14))

    num_patches = cam.shape[0]
    grid_size = int(math.isqrt(num_patches))
    cam = cam.reshape(grid_size, grid_size)

    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    return cam


def _apply_colormap(cam: np.ndarray) -> np.ndarray:
    """Applies a JET colourmap to a normalised [0, 1] heatmap (no matplotlib needed)."""
    cam = np.clip(cam, 0.0, 1.0)

    xs = np.array([0.0, 0.125, 0.375, 0.5, 0.625, 0.875, 1.0])
    b_ys = np.array([0.56, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    g_ys = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    r_ys = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.56])

    r = np.interp(cam, xs, r_ys)
    g = np.interp(cam, xs, g_ys)
    b = np.interp(cam, xs, b_ys)

    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def generate_gradcam_overlay(
    model,
    processor,
    image: Image.Image,
    device: torch.device,
    target_class: Optional[int] = None,
    alpha: float = 0.5,
) -> str:
    """
    Generates a GradCAM heatmap overlay and returns it as a base64-encoded JPEG.

    Args:
        model:        HuggingFace ViTForImageClassification model.
        processor:    The corresponding AutoImageProcessor.
        image:        PIL Image (RGB).
        device:       torch.device to run on.
        target_class: Class index (None = predicted class).
        alpha:        Heatmap overlay opacity (0 = invisible, 1 = opaque).

    Returns:
        Base64-encoded JPEG string of the overlay image.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    cam = compute_gradcam(model, processor, image, device, target_class)

    img_w, img_h = image.size
    cam_pil = Image.fromarray((cam * 255).astype(np.uint8), mode="L")
    cam_pil = cam_pil.resize((img_w, img_h), resample=Image.BICUBIC)
    cam_upsampled = np.array(cam_pil).astype(np.float32) / 255.0

    heatmap_rgb = _apply_colormap(cam_upsampled)
    heatmap_pil = Image.fromarray(heatmap_rgb, mode="RGB")

    original_arr = np.array(image).astype(np.float32)
    heatmap_arr = np.array(heatmap_pil).astype(np.float32)
    blended = (1 - alpha) * original_arr + alpha * heatmap_arr
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    blended_pil = Image.fromarray(blended, mode="RGB")

    buffered = io.BytesIO()
    blended_pil.save(buffered, format="JPEG", quality=85)
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return b64_str
