"""
Meeting Intelligence System — Speaker Diarization
=================================================
Identifies 'who spoke when' using PyAnnote Audio.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from config.settings import settings
from src.models.schemas import SpeakerSegment
from src.utils.device_manager import device_manager

logger = logging.getLogger(__name__)


class SpeakerDiarizer:
    """Wrapper for PyAnnote speaker diarization pipeline."""

    def __init__(self) -> None:
        self.config = settings.diarization
        self.device = device_manager.get_device(self.config.device)
        self.pipeline: Optional[Any] = None

    def load_model(self) -> None:
        """Load the PyAnnote pipeline into memory."""
        if self.pipeline is not None:
            return

        if not self.config.hf_token:
            raise ValueError(
                "HuggingFace token (HF_TOKEN) is required for PyAnnote diarization. "
                "Ensure you have accepted the user conditions at "
                "https://huggingface.co/pyannote/speaker-diarization-3.1"
            )

        logger.info(
            "Loading PyAnnote Diarization %s (Device: %s)",
            self.config.model_name,
            self.device.type,
        )

        try:
            from pyannote.audio import Pipeline
            
            # Load the pipeline from HF hub
            self.pipeline = Pipeline.from_pretrained(
                self.config.model_name,
                use_auth_token=self.config.hf_token,
            )
            
            if self.pipeline is None:
                raise RuntimeError("Failed to load PyAnnote pipeline (returned None)")

            # Move to target device
            self.pipeline.to(self.device)
            logger.info("PyAnnote pipeline loaded and moved to %s", self.device.type)

        except Exception as e:
            logger.error("Failed to load PyAnnote pipeline: %s", e)
            raise RuntimeError(f"Diarizer initialization failed: {e}") from e

    def diarize(
        self,
        audio_path: Path | str,
        num_speakers: Optional[int] = None,
    ) -> list[SpeakerSegment]:
        """
        Run speaker diarization on an audio file.

        Parameters
        ----------
        audio_path : Path | str
            Path to the 16kHz mono WAV file.
        num_speakers : int, optional
            Provide if the exact number of speakers is known in advance.

        Returns
        -------
        list[SpeakerSegment]
            Chronological list of speaker turns with timestamps.
        """
        self.load_model()
        
        audio_path_str = str(audio_path)
        logger.info("Starting diarization for %s", audio_path_str)

        # Build pipeline arguments
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        else:
            if self.config.min_speakers:
                kwargs["min_speakers"] = self.config.min_speakers
            if self.config.max_speakers:
                kwargs["max_speakers"] = self.config.max_speakers

        # Run inference
        diarization = self.pipeline(audio_path_str, **kwargs)

        # Extract segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment = SpeakerSegment(
                speaker=speaker,
                start=turn.start,
                end=turn.end,
            )
            segments.append(segment)
            
        logger.info("Diarization complete: found %d segments", len(segments))
        
        # Sort chronologically just to be safe
        segments.sort(key=lambda x: x.start)
        return segments

    def unload_model(self) -> None:
        """Free GPU memory if needed."""
        self.pipeline = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("PyAnnote pipeline unloaded from VRAM")
