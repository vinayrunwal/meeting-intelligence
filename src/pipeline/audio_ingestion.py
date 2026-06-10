"""
Meeting Intelligence System — Audio Ingestion Pipeline
======================================================
Handles receiving, validating, and preprocessing audio inputs.
Converts everything to 16kHz mono WAV for downstream ML models.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from config.settings import settings
from src.models.schemas import AudioMetadata
from src.utils.audio_utils import (
    AudioProcessingError,
    convert_to_wav,
    get_audio_info,
)

logger = logging.getLogger(__name__)


class AudioIngestionPipeline:
    """Pipeline for ingesting and standardizing audio files."""

    def __init__(self) -> None:
        self.config = settings.audio
        self.upload_dir = settings.UPLOAD_DIR
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def process_file(
        self,
        file_path: Path,
        job_id: Optional[str] = None,
    ) -> tuple[AudioMetadata, Path]:
        """
        Process an uploaded audio file.
        Validates the file, extracts metadata, and converts to standard WAV.

        Parameters
        ----------
        file_path : Path
            Path to the uploaded file.
        job_id : str, optional
            Associated job ID for organizing output files.

        Returns
        -------
        tuple[AudioMetadata, Path]
            The extracted metadata and the path to the standard WAV file.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        job_id = job_id or uuid.uuid4().hex[:12]
        logger.info("Ingesting audio for job %s: %s", job_id, file_path.name)

        # 1. Get raw info and validate
        info = get_audio_info(file_path)
        self._validate_audio(file_path, info)

        # 2. Convert to standard format (16kHz mono WAV)
        job_dir = self.upload_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        standard_wav_path = job_dir / "standardized.wav"
        
        if (
            info["format_name"] == "wav"
            and info["sample_rate"] == self.config.target_sample_rate
            and info["channels"] == self.config.target_channels
        ):
            # Already correct format, just copy
            shutil.copy2(file_path, standard_wav_path)
            logger.info("File already in standard format, copied to %s", standard_wav_path)
        else:
            # Convert using ffmpeg
            convert_to_wav(
                input_path=file_path,
                output_path=standard_wav_path,
                sample_rate=self.config.target_sample_rate,
                channels=self.config.target_channels,
            )

        # 3. Create metadata object
        metadata = AudioMetadata(
            filename=file_path.name,
            duration_seconds=info["duration"],
            sample_rate=self.config.target_sample_rate,
            channels=self.config.target_channels,
            file_size_bytes=standard_wav_path.stat().st_size,
            format="wav",
        )

        logger.info(
            "Ingestion complete for %s (%.1fs)", 
            file_path.name, metadata.duration_seconds
        )
        return metadata, standard_wav_path

    def _validate_audio(self, file_path: Path, info: dict) -> None:
        """Validate audio constraints from settings."""
        # Check size
        size_mb = info["file_size"] / (1024 * 1024)
        if size_mb > self.config.max_upload_size_mb:
            raise AudioProcessingError(
                f"File too large: {size_mb:.1f}MB > {self.config.max_upload_size_mb}MB"
            )

        # Check duration
        duration_min = info["duration"] / 60
        if duration_min > self.config.max_duration_minutes:
            raise AudioProcessingError(
                f"Audio too long: {duration_min:.1f}m > {self.config.max_duration_minutes}m"
            )
        
        if duration_min < 0.05: # Less than 3 seconds
             raise AudioProcessingError("Audio too short (must be > 3 seconds)")

        # Check extension
        ext = file_path.suffix.lower()
        if ext not in self.config.allowed_extensions:
            logger.warning(
                "Unusual extension %s. FFprobe detected format: %s", 
                ext, info["format_name"]
            )
