"""
Decision-Fusion Engine — combines outputs from the Metadata Module and
the Visual Classifier into a single AI-generated probability.

Strategies:
    1. WeightedAverageFusion  — weighted linear combination with accuracy scaling.
    2. ConservativeThresholdFusion — flags AI only when both modules agree (AND-gate).
    3. BayesianFusion — treats each module as independent evidence and applies Bayes' rule.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from PIL import Image


def crop_face_region(image: Image.Image, bbox: list[float] | tuple[float, float, float, float], padding: float = 0.3) -> Image.Image:
    """
    Crops a face bounding box from an image with relative padding, clamped to image boundaries.

    Args:
        image: PIL Image object.
        bbox: Bounding box coordinates [x1, y1, x2, y2].
        padding: Padding factor (e.g. 0.3 for 30%).

    Returns:
        Cropped face PIL Image.
    """
    w, h = image.size
    x1, y1, x2, y2 = bbox
    
    box_w = x2 - x1
    box_h = y2 - y1
    
    pad_w = box_w * padding
    pad_h = box_h * padding
    
    new_x1 = max(0, x1 - pad_w)
    new_y1 = max(0, y1 - pad_h)
    new_x2 = min(w, x2 + pad_w)
    new_y2 = min(h, y2 + pad_h)
    
    return image.crop((new_x1, new_y1, new_x2, new_y2))


@dataclass
class FusionResult:
    """Output of every fusion strategy."""
    ai_probability: float
    is_ai: bool
    formula_name: str
    explanation: dict[str, Any] = field(default_factory=dict)


def extract_visual_ai_probability(visual_result: dict) -> float:
    """
    Normalises VisualClassifier.predict() output into a float in [0, 1]
    representing P(AI-generated).

    Prefers all_scores["AI-generated"] for a direct probability;
    falls back to deriving it from the top-level prediction/confidence.
    """
    all_scores = visual_result.get("all_scores", {})

    ai_score = all_scores.get("AI-generated")
    if ai_score is not None:
        return float(ai_score)

    prediction = visual_result.get("prediction", "")
    confidence = float(visual_result.get("confidence", 0.5))

    if "ai" in prediction.lower() or "generated" in prediction.lower():
        return confidence
    else:
        return 1.0 - confidence


class FusionStrategy(ABC):
    """Base class for all decision-fusion strategies."""

    @abstractmethod
    def fuse(
        self,
        metadata_ai_prob: float,
        visual_ai_prob: float,
        cropped_visual_ai_prob: float | None = None,
    ) -> FusionResult:
        """
        Combine AI-generated probabilities into one decision.

        Args:
            metadata_ai_prob:       P(AI) from the metadata module (0-1).
            visual_ai_prob:         P(AI) from the visual classifier (0-1).
            cropped_visual_ai_prob: Optional P(AI) from the cropped face classifier.
        """


class WeightedAverageFusion(FusionStrategy):
    """
    Weighted average of metadata and visual AI probabilities.
    If a cropped face probability is provided, the visual component uses
    max(visual_prob, cropped_visual_prob).

    Args:
        w_meta: Weight for metadata module (default 0.3).
        w_visual: Weight for visual module (default 0.7).
        decision_threshold: Score above which image is classified AI (default 0.55).
        meta_accuracy: Scale factor for metadata weight (default 0.70).
        visual_accuracy: Scale factor for visual weight (default 0.83).
    """

    def __init__(
        self,
        w_meta: float = 0.3,
        w_visual: float = 0.7,
        decision_threshold: float = 0.55,
        meta_accuracy: float | None = 0.70,
        visual_accuracy: float | None = 0.83,
    ):
        self.w_meta = w_meta
        self.w_visual = w_visual
        self.decision_threshold = decision_threshold
        self.meta_accuracy = meta_accuracy
        self.visual_accuracy = visual_accuracy

    def fuse(
        self,
        metadata_ai_prob: float,
        visual_ai_prob: float,
        cropped_visual_ai_prob: float | None = None,
    ) -> FusionResult:
        has_crop = (cropped_visual_ai_prob is not None)
        if has_crop:
            effective_visual_ai_prob = max(visual_ai_prob, cropped_visual_ai_prob)
        else:
            effective_visual_ai_prob = visual_ai_prob

        eff_w_meta = self.w_meta * (self.meta_accuracy if self.meta_accuracy is not None else 1.0)
        eff_w_visual = self.w_visual * (self.visual_accuracy if self.visual_accuracy is not None else 1.0)

        total = eff_w_meta + eff_w_visual
        if total > 0:
            eff_w_meta_norm = eff_w_meta / total
            eff_w_visual_norm = eff_w_visual / total
        else:
            eff_w_meta_norm = 0.0
            eff_w_visual_norm = 0.0

        combined = eff_w_meta_norm * metadata_ai_prob + eff_w_visual_norm * effective_visual_ai_prob
        combined = max(0.0, min(1.0, combined))

        return FusionResult(
            ai_probability=round(combined, 4),
            is_ai=combined >= self.decision_threshold,
            formula_name="Weighted Average",
            explanation={
                "w_meta_nominal": self.w_meta,
                "w_visual_nominal": self.w_visual,
                "w_meta_effective": round(eff_w_meta_norm, 4),
                "w_visual_effective": round(eff_w_visual_norm, 4),
                "meta_accuracy": self.meta_accuracy,
                "visual_accuracy": self.visual_accuracy,
                "metadata_ai_prob": round(metadata_ai_prob, 4),
                "visual_ai_prob_raw": round(visual_ai_prob, 4),
                "cropped_visual_ai_prob": round(cropped_visual_ai_prob, 4) if has_crop else None,
                "visual_ai_prob_effective": round(effective_visual_ai_prob, 4),
                "combined_score": round(combined, 4),
                "decision_threshold": self.decision_threshold,
            },
        )


class ConservativeThresholdFusion(FusionStrategy):
    """
    Flags AI only when both modules agree above their thresholds (AND-gate).
    This avoids false positives from metadata's tendency to score 0.99 whenever
    any AI marker (e.g. a C2PA tag) is present.

    Args:
        meta_threshold: Metadata module must exceed this (default 0.70).
        visual_threshold: Visual classifier must exceed this (default 0.65).
    """

    def __init__(
        self,
        meta_threshold: float = 0.70,
        visual_threshold: float = 0.65,
    ):
        self.meta_threshold = meta_threshold
        self.visual_threshold = visual_threshold

    def fuse(
        self,
        metadata_ai_prob: float,
        visual_ai_prob: float,
        cropped_visual_ai_prob: float | None = None,
    ) -> FusionResult:
        meta_pass = metadata_ai_prob >= self.meta_threshold
        visual_pass = visual_ai_prob >= self.visual_threshold

        both_pass = meta_pass and visual_pass

        if both_pass:
            combined = min(metadata_ai_prob, visual_ai_prob)
        else:
            combined = min(metadata_ai_prob, visual_ai_prob) * 0.5

        combined = max(0.0, min(1.0, combined))

        return FusionResult(
            ai_probability=round(combined, 4),
            is_ai=both_pass,
            formula_name="Conservative Threshold (AND-gate)",
            explanation={
                "meta_threshold": self.meta_threshold,
                "visual_threshold": self.visual_threshold,
                "metadata_ai_prob": round(metadata_ai_prob, 4),
                "visual_ai_prob": round(visual_ai_prob, 4),
                "meta_exceeds_threshold": meta_pass,
                "visual_exceeds_threshold": visual_pass,
                "both_agree": both_pass,
                "combined_score": round(combined, 4),
            },
        )


class BayesianFusion(FusionStrategy):
    """
    Bayesian evidence fusion treating each module as an independent likelihood.

    Uses Bayes' rule with a configurable prior:
        L_ai  = p1 * p2
        L_real = (1-p1) * (1-p2)
        P(AI | evidence) = (L_ai * prior) / (L_ai * prior + L_real * (1 - prior))

    Args:
        prior: Prior P(AI) for any image (default 0.5).
        decision_threshold: Posterior above which we classify as AI (default 0.55).
    """

    def __init__(
        self,
        prior: float = 0.5,
        decision_threshold: float = 0.55,
    ):
        self.prior = prior
        self.decision_threshold = decision_threshold

    def fuse(
        self,
        metadata_ai_prob: float,
        visual_ai_prob: float,
        cropped_visual_ai_prob: float | None = None,
    ) -> FusionResult:
        eps = 1e-9
        p1 = max(eps, min(1 - eps, metadata_ai_prob))
        p2 = max(eps, min(1 - eps, visual_ai_prob))

        likelihood_ai = p1 * p2
        likelihood_real = (1 - p1) * (1 - p2)

        numerator = likelihood_ai * self.prior
        denominator = numerator + likelihood_real * (1 - self.prior)

        posterior = numerator / denominator
        posterior = max(0.0, min(1.0, posterior))

        return FusionResult(
            ai_probability=round(posterior, 4),
            is_ai=posterior >= self.decision_threshold,
            formula_name="Bayesian Fusion",
            explanation={
                "prior": self.prior,
                "metadata_ai_prob": round(metadata_ai_prob, 4),
                "visual_ai_prob": round(visual_ai_prob, 4),
                "likelihood_ai": round(likelihood_ai, 6),
                "likelihood_real": round(likelihood_real, 6),
                "posterior": round(posterior, 4),
                "decision_threshold": self.decision_threshold,
            },
        )


AVAILABLE_STRATEGIES: dict[str, type[FusionStrategy]] = {
    "weighted_average": WeightedAverageFusion,
    "conservative_threshold": ConservativeThresholdFusion,
    "bayesian": BayesianFusion,
}


def get_fusion_strategy(name: str, **kwargs) -> FusionStrategy:
    """
    Factory that returns a configured FusionStrategy by name.

    Args:
        name:     One of "weighted_average", "conservative_threshold", "bayesian".
        **kwargs: Forwarded to the strategy's __init__.

    Raises:
        ValueError: If name is not recognised.
    """
    cls = AVAILABLE_STRATEGIES.get(name)
    if cls is None:
        valid = ", ".join(sorted(AVAILABLE_STRATEGIES))
        raise ValueError(
            f"Unknown fusion strategy '{name}'. Choose from: {valid}"
        )
    return cls(**kwargs)
