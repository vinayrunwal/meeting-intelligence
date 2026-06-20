"""
Meeting Intelligence System — Speaker Diarization
=================================================
Identifies 'who spoke when' using PyAnnote Audio.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Any

import torch
from torch.torch_version import TorchVersion

from config.settings import settings
from src.models.schemas import SpeakerSegment
from src.utils.device_manager import device_manager

logger = logging.getLogger(__name__)


@contextmanager
def _trusted_torch_checkpoints():
    """Allow trusted PyAnnote checkpoints to load under PyTorch 2.6+."""
    original_load = torch.load

    def trusted_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([TorchVersion])
    torch.load = trusted_load
    try:
        yield
    finally:
        torch.load = original_load


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
            import pyannote.audio.core.model as pyannote_model
            import pyannote.audio.core.pipeline as pyannote_pipeline
            import pyannote.audio.pipelines.speaker_verification as pyannote_speaker_verification
            from huggingface_hub import hf_hub_download

            def hf_hub_download_compat(*args, use_auth_token=None, token=None, **kwargs):
                if token is None:
                    token = use_auth_token
                return hf_hub_download(*args, token=token, **kwargs)

            pyannote_model.hf_hub_download = hf_hub_download_compat
            pyannote_pipeline.hf_hub_download = hf_hub_download_compat
            pyannote_speaker_verification.hf_hub_download = hf_hub_download_compat

            with _trusted_torch_checkpoints():
                # Load the pipeline from HF hub
                try:
                    self.pipeline = Pipeline.from_pretrained(
                        self.config.model_name,
                        token=self.config.hf_token,
                        cache_dir=str(settings.MODEL_CACHE_DIR / "pyannote"),
                    )
                except TypeError:
                    self.pipeline = Pipeline.from_pretrained(
                        self.config.model_name,
                        use_auth_token=self.config.hf_token,
                    )

                if self.pipeline is None:
                    raise RuntimeError("Failed to load PyAnnote pipeline (returned None)")

                # Move to target device while PyAnnote may still lazily load weights.
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
