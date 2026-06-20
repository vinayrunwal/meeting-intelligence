"""
Meeting Intelligence System — Intent Classifier
===============================================
Zero-shot text classification to determine the intent behind a speaker's turn.
"""

from __future__ import annotations

import logging
from typing import Optional, Any

from config.settings import settings
from src.models.schemas import IntentResult
from src.utils.device_manager import device_manager

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Wrapper for zero-shot intent classification pipeline."""

    def __init__(self) -> None:
        self.config = settings.intent
        self.device = device_manager.get_device(self.config.device)
        self.classifier: Optional[Any] = None

    def load_model(self) -> None:
        """Load the HuggingFace zero-shot pipeline into memory."""
        if self.classifier is not None:
            return

        logger.info(
            "Loading Intent Classifier %s (Device: %s)",
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
                "zero-shot-classification",
                model=self.config.model_name,
                device=device_idx,
            )
            logger.info("Intent classifier loaded successfully")

        except Exception as e:
            logger.error("Failed to load Intent Classifier: %s", e)
            raise RuntimeError(f"Intent Classifier initialization failed: {e}") from e

    def classify(self, text: str) -> IntentResult:
        """
        Classify the intent of a single text segment.

        Parameters
        ----------
        text : str
            The text to classify.

        Returns
        -------
        IntentResult
            The primary intent, confidence, and all other scores.
        """
        self.load_model()

        if not text.strip():
            return IntentResult(label="statement", confidence=1.0, all_scores={"statement": 1.0})

        try:
            result = self.classifier(
                text,
                candidate_labels=list(self.config.candidate_labels),
                multi_label=False,
            )
            
            # result is a dict with 'labels' and 'scores' arrays
            labels = result["labels"]
            scores = result["scores"]
            
            all_scores = dict(zip(labels, scores))
            top_label = labels[0]
            confidence = scores[0]

            return IntentResult(
                label=top_label,
                confidence=confidence,
                all_scores=all_scores,
            )
            
        except Exception as e:
            logger.warning("Intent classification failed for text '%s...': %s", text[:20], e)
            return IntentResult(label="statement", confidence=0.0, all_scores={})

    def classify_batch(self, texts: list[str]) -> list[IntentResult]:
        """
        Classify the intent of multiple text segments efficiently.

        Parameters
        ----------
        texts : list[str]
            List of texts to classify.

        Returns
        -------
        list[IntentResult]
            List of classification results corresponding to the inputs.
        """
        self.load_model()
        
        if not texts:
            return []

        safe_texts = [t if t.strip() else "statement" for t in texts]
        
        try:
            # Batch inference
            batch_results = self.classifier(
                safe_texts,
                candidate_labels=list(self.config.candidate_labels),
                multi_label=False,
                batch_size=self.config.batch_size,
            )
            
            final_results = []
            for result in batch_results:
                labels = result["labels"]
                scores = result["scores"]
                
                all_scores = dict(zip(labels, scores))
                
                final_results.append(
                    IntentResult(
                        label=labels[0],
                        confidence=scores[0],
                        all_scores=all_scores,
                    )
                )
            return final_results
            
        except Exception as e:
            logger.error("Batch intent classification failed: %s", e)
            # Fallback
            return [
                IntentResult(label="statement", confidence=0.0, all_scores={})
                for _ in texts
            ]

    def unload_model(self) -> None:
        """Free GPU memory if needed."""
        self.classifier = None
        if self.device.type == "cuda":
            import torch
            torch.cuda.empty_cache()
            logger.info("Intent classifier unloaded from VRAM")
