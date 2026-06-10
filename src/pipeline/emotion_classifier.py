"""
Meeting Intelligence System — Emotion Classifier
================================================
Classifies emotions from text segments using HuggingFace Transformers.
"""

from __future__ import annotations

import logging
from typing import Optional

from config.settings import settings
from src.models.schemas import EmotionResult
from src.utils.device_manager import device_manager

logger = logging.getLogger(__name__)


class EmotionClassifier:
    """Wrapper for text-based emotion classification pipeline."""

    def __init__(self) -> None:
        self.config = settings.emotion
        self.device = device_manager.get_device(self.config.device)
        self.classifier: Optional[Any] = None

    def load_model(self) -> None:
        """Load the HuggingFace pipeline into memory."""
        if self.classifier is not None:
            return

        logger.info(
            "Loading Emotion Classifier %s (Device: %s)",
            self.config.model_name,
            self.device.type,
        )

        try:
            from transformers import pipeline

            # Convert device to index for pipeline (-1 for CPU, 0+ for GPU)
            device_idx = -1
            if self.device.type == "cuda":
                device_idx = self.device.index if self.device.index is not None else 0
            elif self.device.type == "mps":
                device_idx = self.device

            self.classifier = pipeline(
                "text-classification",
                model=self.config.model_name,
                device=device_idx,
                top_k=None,  # Return scores for all classes
            )
            logger.info("Emotion classifier loaded successfully")

        except Exception as e:
            logger.error("Failed to load Emotion Classifier: %s", e)
            raise RuntimeError(f"Emotion Classifier initialization failed: {e}") from e

    def classify(self, text: str) -> EmotionResult:
        """
        Classify the emotion of a single text segment.

        Parameters
        ----------
        text : str
            The text to classify.

        Returns
        -------
        EmotionResult
            The primary emotion, confidence, and all other scores.
        """
        self.load_model()

        if not text.strip():
            return EmotionResult(label="neutral", confidence=1.0, all_scores={"neutral": 1.0})

        try:
            # The pipeline with top_k=None returns a list of lists of dicts
            results = self.classifier(text[: self.config.max_length])
            
            # Extract scores
            scores_list = results[0]
            all_scores = {item["label"].lower(): item["score"] for item in scores_list}
            
            # Find the top emotion
            top_label = max(all_scores, key=all_scores.get)
            confidence = all_scores[top_label]

            return EmotionResult(
                label=top_label,
                confidence=confidence,
                all_scores=all_scores,
            )
            
        except Exception as e:
            logger.warning("Emotion classification failed for text '%s...': %s", text[:20], e)
            return EmotionResult(label="neutral", confidence=0.0, all_scores={})

    def classify_batch(self, texts: list[str]) -> list[EmotionResult]:
        """
        Classify the emotion of multiple text segments efficiently.

        Parameters
        ----------
        texts : list[str]
            List of texts to classify.

        Returns
        -------
        list[EmotionResult]
            List of classification results corresponding to the inputs.
        """
        self.load_model()
        
        if not texts:
            return []

        # Truncate texts to max length
        safe_texts = [t[: self.config.max_length] if t.strip() else "neutral" for t in texts]
        
        try:
            # Batch inference
            batch_results = self.classifier(safe_texts, batch_size=self.config.batch_size)
            
            final_results = []
            for results in batch_results:
                all_scores = {item["label"].lower(): item["score"] for item in results}
                top_label = max(all_scores, key=all_scores.get)
                
                final_results.append(
                    EmotionResult(
                        label=top_label,
                        confidence=all_scores[top_label],
                        all_scores=all_scores,
                    )
                )
            return final_results
            
        except Exception as e:
            logger.error("Batch emotion classification failed: %s", e)
            # Fallback to neutral
            return [
                EmotionResult(label="neutral", confidence=0.0, all_scores={})
                for _ in texts
            ]

    def unload_model(self) -> None:
        """Free GPU memory if needed."""
        self.classifier = None
        if self.device.type == "cuda":
            import torch
            torch.cuda.empty_cache()
            logger.info("Emotion classifier unloaded from VRAM")
