#!/usr/bin/env python3
"""
Meeting Intelligence System — Emotion Evaluator
===============================================
Computes F1 Score and classification metrics for the Emotion Classifier.
"""

import logging
from typing import List, Dict

from sklearn.metrics import classification_report, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_emotion_classification(
    y_true: List[str], 
    y_pred: List[str]
) -> Dict[str, float | str]:
    """
    Calculate F1 scores and other classification metrics for emotion predictions.

    Parameters
    ----------
    y_true : List[str]
        Ground truth emotion labels.
    y_pred : List[str]
        Predicted emotion labels.

    Returns
    -------
    dict
        Metrics including weighted F1, macro F1, and full report string.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Length of true labels and predictions must match.")

    if not y_true:
        return {}

    logger.info(f"Evaluating Emotion Classification on {len(y_true)} samples...")

    # Compute F1 scores
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Generate full report
    report = classification_report(y_true, y_pred, zero_division=0)

    metrics = {
        "f1_weighted": float(f1_weighted),
        "f1_macro": float(f1_macro),
        "classification_report": report,
    }

    logger.info(f"Weighted F1: {f1_weighted:.4f}")
    return metrics
