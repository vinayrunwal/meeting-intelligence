#!/usr/bin/env python3
"""
Meeting Intelligence System — Model Downloader
==============================================
Pre-downloads HuggingFace and Whisper models into the local cache.
Useful for Docker builds or offline deployments.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def download_whisper(model_size: str) -> None:
    """Download faster-whisper model."""
    logger.info("Downloading Whisper model: %s", model_size)
    from faster_whisper import download_model
    
    download_root = str(settings.MODEL_CACHE_DIR)
    os.makedirs(download_root, exist_ok=True)
    
    try:
        model_path = download_model(model_size, cache_dir=download_root)
        logger.info("Whisper model saved to: %s", model_path)
    except Exception as e:
        logger.error("Failed to download Whisper model: %s", e)


def download_pyannote(model_name: str, hf_token: str) -> None:
    """Download PyAnnote pipeline."""
    if not hf_token:
        logger.warning("Skipping PyAnnote: HF_TOKEN not provided")
        return
        
    logger.info("Downloading PyAnnote model: %s", model_name)
    from pyannote.audio import Pipeline
    
    try:
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
        if pipeline:
            logger.info("PyAnnote model downloaded successfully")
    except Exception as e:
        logger.error("Failed to download PyAnnote model: %s", e)


def download_huggingface_model(model_name: str, task: str = "text-classification") -> None:
    """Download standard HuggingFace transformer model."""
    logger.info("Downloading %s model: %s", task, model_name)
    from transformers import pipeline
    
    try:
        classifier = pipeline(task, model=model_name)
        if classifier:
            logger.info("%s model downloaded successfully", task.capitalize())
    except Exception as e:
        logger.error("Failed to download %s model: %s", task, e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ML models for offline use.")
    parser.add_argument("--all", action="store_true", help="Download all default models")
    parser.add_argument("--whisper", action="store_true", help="Download Whisper model")
    parser.add_argument("--diarization", action="store_true", help="Download PyAnnote model")
    parser.add_argument("--emotion", action="store_true", help="Download Emotion model")
    parser.add_argument("--intent", action="store_true", help="Download Intent model")
    parser.add_argument("--debrief", action="store_true", help="Download Debrief model")
    
    args = parser.parse_args()
    
    if args.all or args.whisper:
        download_whisper(settings.whisper.model_size)
        
    if args.all or args.diarization:
        download_pyannote(settings.diarization.model_name, settings.diarization.hf_token)
        
    if args.all or args.emotion:
        download_huggingface_model(settings.emotion.model_name, "text-classification")
        
    if args.all or args.intent:
        download_huggingface_model(settings.intent.model_name, "zero-shot-classification")
        
    if args.all or args.debrief:
        download_huggingface_model(settings.debrief.model_name, "text2text-generation")
        
    if not any([args.all, args.whisper, args.diarization, args.emotion, args.intent, args.debrief]):
        parser.print_help()


if __name__ == "__main__":
    main()
