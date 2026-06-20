#!/usr/bin/env python3
"""
Meeting Intelligence System — Benchmark Runner
==============================================
Main entry point to execute all benchmarks and generate a Markdown report.
"""

import logging
from datetime import datetime
from pathlib import Path

from benchmarks.emotion_evaluator import evaluate_emotion_classification
from benchmarks.wer_evaluator import evaluate_wer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_report(results: dict, output_path: Path):
    """Generate a Markdown report from benchmark results."""
    report = [
        "# Meeting Intelligence System — Benchmark Report",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. Speech Recognition (ASR) Accuracy",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Word Error Rate (WER) | {results.get('asr', {}).get('wer', 0):.2%} |",
        f"| Match Error Rate (MER)| {results.get('asr', {}).get('mer', 0):.2%} |",
        "",
        "## 2. Emotion Classification",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Weighted F1 | {results.get('emotion', {}).get('f1_weighted', 0):.4f} |",
        f"| Macro F1 | {results.get('emotion', {}).get('f1_macro', 0):.4f} |",
        "",
        "### Detailed Emotion Report",
        "```text",
        results.get('emotion', {}).get('classification_report', 'N/A'),
        "```",
    ]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(report))
        
    logger.info(f"Report saved to {output_path}")


def main():
    logger.info("Starting Benchmark Suite...")
    
    # 1. Dummy Data for demonstration
    # In a real scenario, this would load from a validation dataset
    dummy_refs = ["hello world this is a test", "the system is working perfectly"]
    dummy_hyps = ["hello world this is test", "the system working perfectly"]
    
    dummy_emo_true = ["joy", "neutral", "anger", "joy"]
    dummy_emo_pred = ["joy", "neutral", "sadness", "joy"]
    
    # 2. Run Evaluators
    asr_results = evaluate_wer(dummy_refs, dummy_hyps)
    emotion_results = evaluate_emotion_classification(dummy_emo_true, dummy_emo_pred)
    
    # 3. Combine Results
    results = {
        "asr": asr_results,
        "emotion": emotion_results,
    }
    
    # 4. Generate Report
    report_path = Path("benchmarks/reports/benchmark_report.md")
    generate_report(results, report_path)


if __name__ == "__main__":
    main()
