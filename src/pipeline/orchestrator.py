"""
Meeting Intelligence System — Pipeline Orchestrator
===================================================
Coordinates the full execution from Audio Ingestion to Meeting Debrief.
Manages state, handles errors, and reports progress.
"""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path
from typing import Optional

from src.models.schemas import (
    AudioMetadata,
    EnrichedSegment,
    JobResult,
    JobState,
    MeetingDebrief,
)
from src.pipeline.asr_engine import WhisperASREngine
from src.pipeline.audio_ingestion import AudioIngestionPipeline
from src.pipeline.diarization import SpeakerDiarizer
from src.pipeline.transcript_aligner import TranscriptAligner
from src.pipeline.emotion_classifier import EmotionClassifier
from src.pipeline.intent_classifier import IntentClassifier
from src.pipeline.debrief_generator import DebriefGenerator
from src.utils.job_store import job_store

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the entire ML pipeline execution."""

    def __init__(self) -> None:
        self.ingestion = AudioIngestionPipeline()
        # Initialize other components lazily to save memory
        self.asr: Optional[WhisperASREngine] = None
        self.diarizer: Optional[SpeakerDiarizer] = None
        self.aligner = TranscriptAligner()
        self.emotion_classifier: Optional[EmotionClassifier] = None
        self.intent_classifier: Optional[IntentClassifier] = None
        self.debrief_generator: Optional[DebriefGenerator] = None

    def process_job(self, job_id: str, file_path: Path) -> None:
        """
        Run the full processing pipeline for a given job.
        
        Parameters
        ----------
        job_id : str
            The ID of the job being processed.
        file_path : Path
            The path to the uploaded audio file.
        """
        start_time = time.time()
        audio_metadata: Optional[AudioMetadata] = None
        aligned_segments = []
        enriched_segments = []
        debrief: Optional[MeetingDebrief] = None

        try:
            logger.info("Starting pipeline for job %s", job_id)
            job_store.update_progress(
                job_id, JobState.PROCESSING, "ingestion", 5.0, 
                "Ingesting and standardizing audio"
            )

            # -------------------------------------------------------------
            # Step 1.1: Audio Ingestion & Preprocessing
            # -------------------------------------------------------------
            audio_metadata, wav_path = self.ingestion.process_file(file_path, job_id)
            job_store.set_audio_metadata(job_id, audio_metadata)
            
            job_store.update_progress(
                job_id, JobState.TRANSCRIBING, "transcription", 15.0,
                f"Transcribing {audio_metadata.duration_seconds:.1f}s of audio"
            )

            # -------------------------------------------------------------
            # Step 1.2: ASR (Whisper)
            # -------------------------------------------------------------
            if self.asr is None:
                self.asr = WhisperASREngine()
                
            transcript_segments = self.asr.transcribe(wav_path)
            
            job_store.update_progress(
                job_id, JobState.DIARIZING, "diarization", 40.0,
                "Identifying speakers"
            )

            # -------------------------------------------------------------
            # Step 1.3: Speaker Diarization (PyAnnote)
            # -------------------------------------------------------------
            # Free ASR memory before loading diarization if on same GPU
            self.asr.unload_model()
            
            if self.diarizer is None:
                self.diarizer = SpeakerDiarizer()
                
            speaker_segments = self.diarizer.diarize(wav_path)
            
            # -------------------------------------------------------------
            # Step 1.4: Alignment
            # -------------------------------------------------------------
            job_store.update_progress(
                job_id, JobState.DIARIZING, "alignment", 60.0,
                "Aligning text with speakers"
            )
            aligned_segments = self.aligner.align(transcript_segments, speaker_segments)
            
            self.diarizer.unload_model()

            # -------------------------------------------------------------
            # Step 2: Emotion & Intent Classification
            # -------------------------------------------------------------
            job_store.update_progress(
                job_id, JobState.CLASSIFYING, "classification", 70.0,
                "Classifying emotion and intent per speaker turn"
            )
            
            if self.emotion_classifier is None:
                self.emotion_classifier = EmotionClassifier()
            if self.intent_classifier is None:
                self.intent_classifier = IntentClassifier()
                
            texts = [seg.text for seg in aligned_segments]
            emotions = self.emotion_classifier.classify_batch(texts)
            intents = self.intent_classifier.classify_batch(texts)
            
            for seg, emotion, intent in zip(aligned_segments, emotions, intents):
                enriched_segments.append(
                    EnrichedSegment(
                        id=seg.id,
                        speaker=seg.speaker,
                        text=seg.text,
                        start=seg.start,
                        end=seg.end,
                        language=seg.language,
                        confidence=seg.confidence,
                        emotion=emotion,
                        intent=intent,
                    )
                )
                
            self.emotion_classifier.unload_model()
            self.intent_classifier.unload_model()

            # -------------------------------------------------------------
            # Step 3: Debrief Generation
            # -------------------------------------------------------------
            job_store.update_progress(
                job_id, JobState.SUMMARIZING, "summarization", 85.0,
                "Generating meeting debrief (decisions, action items, conflicts)"
            )
            
            if self.debrief_generator is None:
                self.debrief_generator = DebriefGenerator()
                
            debrief = self.debrief_generator.generate_debrief(enriched_segments)
            self.debrief_generator.unload_model()

            # -------------------------------------------------------------
            # Completion
            # -------------------------------------------------------------
            processing_time = time.time() - start_time
            
            result = JobResult(
                job_id=job_id,
                state=JobState.COMPLETED,
                audio_metadata=audio_metadata,
                segments=enriched_segments,
                debrief=debrief,
                processing_time_seconds=processing_time,
            )
            
            job_store.store_result(job_id, result)
            logger.info("Pipeline completed successfully for job %s", job_id)

        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            logger.error("Error processing job %s:\n%s", job_id, traceback.format_exc())
            job_store.set_failed(job_id, error_msg)

        finally:
            # Cleanup temporary wav file if we created it
            try:
                if 'wav_path' in locals() and wav_path.exists():
                    wav_path.unlink()
            except Exception as e:
                logger.warning("Failed to clean up temporary wav file: %s", e)
