"""
GradCAM for Vision Transformer (ViT) Models
=============================================
Generates Gradient-weighted Class Activation Mapping heatmaps for
HuggingFace ViT image classification models.

Standard CNN-style GradCAM (hooking into convolutional feature maps) does
NOT work well for ViT because:

  1. LayerNorm normalises each token independently, flattening spatial
     variation in activations and gradients.
  2. ViT has no spatial feature maps — only a sequence of patch tokens.

Instead, this module uses an **Attention-Gradient** approach:

  1. Extract the CLS token's attention weights to each patch from the
     last self-attention layer (these naturally encode spatial focus).
  2. Compute the gradient of the target class logit w.r.t. the last
     encoder layer's hidden states.
  3. Use per-patch gradient magnitude as a class-specific weighting
     factor on the attention map.

  cam_k = attn_CLS→k × ‖∂y_c / ∂h_k‖₂

If gradient computation fails (e.g. on MPS), falls back to pure
attention-based heatmap which still provides useful spatial information.

The resulting 14×14 heatmap (for patch_size=16, image_size=224) is then
upsampled and overlaid on the original image.

Reference: Chefer et al. (2021) "Transformer Interpretability Beyond
Attention Visualization" — simplified to single-layer for efficiency.

Usage
-----
    from src.visual_module.gradcam import generate_gradcam_overlay

    overlay_b64 = generate_gradcam_overlay(
        model=visual_classifier.model,
        processor=visual_classifier.processor,
        image=pil_image,
        device=visual_classifier.device,
    )
"""

from __future__ import annotations

import base64
import io
import math
from typing import Optional

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Core Attention-Gradient CAM computation
# ---------------------------------------------------------------------------

def compute_gradcam(
    model,
    processor,
    image: Image.Image,
    device: torch.device,
    target_class: Optional[int] = None,
) -> np.ndarray:
    """
    Computes an attention-gradient heatmap for a given image and ViT model.

    The method combines two signals:
      - **Attention**: CLS token's attention to each patch in the last layer
        (averaged over heads). This captures the model's spatial focus.
      - **Gradient**: Gradient of the target class logit w.r.t. the attention
        weights of the last layer, which provides class-specific weighting.

    Args:
        model:        HuggingFace ViTForImageClassification model.
        processor:    The corresponding AutoImageProcessor.
        image:        PIL Image (RGB).
        device:       torch.device to run on.
        target_class: Class index to compute gradients for.
                      If None, uses the predicted (argmax) class.

    Returns:
        cam: numpy array of shape (grid_h, grid_w) with values in [0, 1],
             representing the normalised heatmap.
    """
    model.eval()

    # Prepare input
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    cam = None

    try:
        with torch.enable_grad():
            # Forward pass with attention output enabled
            outputs = model(**inputs, output_attentions=True)
            logits = outputs.logits  # shape: (1, num_classes)

            if target_class is None:
                target_class = logits.argmax(dim=-1).item()

            attn_tuple = outputs.attentions
            if attn_tuple is not None and len(attn_tuple) > 0:
                last_attn = attn_tuple[-1]  # (1, heads, seq, seq)
                last_attn.retain_grad()

                model.zero_grad()
                target_logit = logits[0, target_class]
                target_logit.backward()

                if last_attn.grad is not None:
                    # Get CLS token attention and its gradient w.r.t. patch tokens
                    attn = last_attn[0, :, 0, 1:]  # (heads, num_patches)
                    grad = last_attn.grad[0, :, 0, 1:]  # (heads, num_patches)

                    # Compute class-specific GradCAM using positive gradients
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

    # Last resort fallback if everything failed
    if cam is None:
        return np.zeros((14, 14))

    # Reshape to 2D patch grid
    num_patches = cam.shape[0]
    grid_size = int(math.isqrt(num_patches))
    cam = cam.reshape(grid_size, grid_size)

    # Normalise to [0, 1]
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    return cam


# ---------------------------------------------------------------------------
# Heatmap overlay generation
# ---------------------------------------------------------------------------

def _apply_colormap(cam: np.ndarray) -> np.ndarray:
    """
    Applies a standard JET colourmap (blue → cyan → green → yellow → red) to a
    normalised [0, 1] heatmap without requiring matplotlib.

    Args:
        cam: 2D numpy array with values in [0, 1].

    Returns:
        RGB numpy array of shape (H, W, 3) with uint8 values.
    """
    cam = np.clip(cam, 0.0, 1.0)

    # Standard JET colormap control points
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
    Generates a GradCAM heatmap overlay on the original image and returns
    it as a base64-encoded JPEG string.

    Args:
        model:        HuggingFace ViTForImageClassification model.
        processor:    The corresponding AutoImageProcessor.
        image:        PIL Image (RGB).
        device:       torch.device to run on.
        target_class: Class index to compute gradients for (None = predicted).
        alpha:        Opacity of the heatmap overlay (0 = invisible, 1 = opaque).

    Returns:
        Base64-encoded JPEG string of the overlay image.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 1. Compute the attention-gradient heatmap
    cam = compute_gradcam(model, processor, image, device, target_class)

    # 2. Upsample heatmap to original image size using bicubic interpolation
    img_w, img_h = image.size

    cam_pil = Image.fromarray((cam * 255).astype(np.uint8), mode="L")
    cam_pil = cam_pil.resize((img_w, img_h), resample=Image.BICUBIC)
    cam_upsampled = np.array(cam_pil).astype(np.float32) / 255.0

    # 3. Apply JET colourmap to create RGB heatmap
    heatmap_rgb = _apply_colormap(cam_upsampled)
    heatmap_pil = Image.fromarray(heatmap_rgb, mode="RGB")

    # 4. Blend with original image
    original_arr = np.array(image).astype(np.float32)
    heatmap_arr = np.array(heatmap_pil).astype(np.float32)
    blended = (1 - alpha) * original_arr + alpha * heatmap_arr
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    blended_pil = Image.fromarray(blended, mode="RGB")

    # 5. Encode to base64 JPEG
    buffered = io.BytesIO()
    blended_pil.save(buffered, format="JPEG", quality=85)
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return b64_str
