"""
Meeting Intelligence System — ASR Engine
========================================
Automatic Speech Recognition using faster-whisper.
Extracts text with word-level timestamps for diarization alignment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Optional

from faster_whisper import WhisperModel

from config.settings import settings
from src.models.schemas import TranscriptSegment, TranscriptWord
from src.utils.device_manager import device_manager

logger = logging.getLogger(__name__)


class WhisperASREngine:
    """Wrapper for faster-whisper with device optimization."""

    def __init__(self) -> None:
        self.config = settings.whisper
        self.device = device_manager.get_device(self.config.device)
        self.compute_type = device_manager.get_compute_type(self.config.compute_type)
        self.model: Optional[WhisperModel] = None

    def load_model(self) -> None:
        """Load the Whisper model into memory."""
        if self.model is not None:
            return

        logger.info(
            "Loading Whisper %s (Device: %s, Compute: %s)",
            self.config.model_size,
            self.device.type,
            self.compute_type,
        )
        
        # faster_whisper expects 'cuda' or 'cpu' as device string
        device_str = "cuda" if self.device.type == "cuda" else "cpu"
        
        # Check memory if using GPU
        if device_str == "cuda":
            # rough estimates based on model size
            mem_req = {
                "large-v3": 4.0,
                "medium": 2.5,
                "small": 1.5,
                "base": 1.0,
                "tiny": 1.0
            }.get(self.config.model_size, 4.0)
            
            device_manager.check_memory(mem_req)

        try:
            self.model = WhisperModel(
                self.config.model_size,
                device=device_str,
                compute_type=self.compute_type,
                # Avoid downloading to arbitrary directories
                download_root=str(settings.MODEL_CACHE_DIR),
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error("Failed to load Whisper model: %s", e)
            raise RuntimeError(f"ASR engine initialization failed: {e}") from e

    def transcribe(
        self, 
        audio_path: Path | str, 
        language_hint: Optional[str] = None
    ) -> list[TranscriptSegment]:
        """
        Transcribe an audio file and return word-level timestamps.

        Parameters
        ----------
        audio_path : Path | str
            Path to the 16kHz mono WAV file.
        language_hint : str, optional
            ISO language code (e.g. 'en', 'fr') to override auto-detection.

        Returns
        -------
        list[TranscriptSegment]
            List of transcribed segments with word-level details.
        """
        self.load_model()
        
        audio_path_str = str(audio_path)
        logger.info("Starting transcription for %s", audio_path_str)

        lang = language_hint or self.config.language

        segments, info = self.model.transcribe(
            audio_path_str,
            beam_size=self.config.beam_size,
            language=lang,
            task=self.config.task,
            vad_filter=self.config.vad_filter,
            word_timestamps=self.config.word_timestamps,
            condition_on_previous_text=self.config.condition_on_previous_text,
            initial_prompt=self.config.initial_prompt,
        )

        logger.info(
            "Detected language '%s' with probability %.2f",
            info.language, info.language_probability
        )

        result_segments = []
        for segment in segments:
            words = []
            if segment.words:
                words = [
                    TranscriptWord(
                        word=w.word,
                        start=w.start,
                        end=w.end,
                        confidence=w.probability,
                    )
                    for w in segment.words
                ]

            ts_segment = TranscriptSegment(
                id=segment.id,
                text=segment.text,
                start=segment.start,
                end=segment.end,
                confidence=segment.avg_logprob,  # Rough proxy for confidence
                language=info.language,
                words=words,
            )
            result_segments.append(ts_segment)
            
            logger.debug(
                "Segment [%.2fs - %.2fs]: %s", 
                segment.start, segment.end, segment.text.strip()
            )

        logger.info("Transcription completed with %d segments", len(result_segments))
        return result_segments

    def unload_model(self) -> None:
        """Free GPU memory if needed."""
        self.model = None
        if self.device.type == "cuda":
            import torch
            torch.cuda.empty_cache()
            logger.info("Whisper model unloaded from VRAM")
