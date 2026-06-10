"""
Meeting Intelligence System — Transcript Aligner
================================================
Aligns Whisper text segments with PyAnnote speaker segments
using temporal overlap maximization.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.models.schemas import (
    AlignedSegment,
    SpeakerSegment,
    TranscriptSegment,
    TranscriptWord,
)

logger = logging.getLogger(__name__)


class TranscriptAligner:
    """Aligns ASR transcripts with speaker diarization labels."""

    def __init__(self) -> None:
        pass

    def align(
        self,
        transcript_segments: list[TranscriptSegment],
        speaker_segments: list[SpeakerSegment],
    ) -> list[AlignedSegment]:
        """
        Align transcript segments to speaker segments.

        The algorithm works at the word level if word timestamps are available,
        otherwise it falls back to segment-level overlap.

        Parameters
        ----------
        transcript_segments : list[TranscriptSegment]
            Output from the ASR engine.
        speaker_segments : list[SpeakerSegment]
            Output from the diarization engine.

        Returns
        -------
        list[AlignedSegment]
            List of text segments assigned to specific speakers.
        """
        if not transcript_segments:
            return []
            
        if not speaker_segments:
            # Fallback if diarization failed/returned empty
            logger.warning("No speaker segments provided. Assigning all to SPEAKER_UNKNOWN.")
            return [
                AlignedSegment(
                    id=ts.id,
                    speaker="SPEAKER_UNKNOWN",
                    text=ts.text,
                    start=ts.start,
                    end=ts.end,
                    language=ts.language,
                    confidence=ts.confidence,
                    words=ts.words,
                )
                for ts in transcript_segments
            ]

        logger.info(
            "Aligning %d transcript segments to %d speaker segments",
            len(transcript_segments),
            len(speaker_segments),
        )

        aligned_results: list[AlignedSegment] = []

        for ts_segment in transcript_segments:
            if ts_segment.words:
                # Word-level alignment (more accurate)
                aligned_segment = self._align_by_words(ts_segment, speaker_segments)
            else:
                # Segment-level alignment (fallback)
                aligned_segment = self._align_by_segment(ts_segment, speaker_segments)
                
            aligned_results.append(aligned_segment)

        # Optional: Merge consecutive segments by the same speaker
        merged_results = self._merge_consecutive_segments(aligned_results)
        
        logger.info("Alignment complete. Yielded %d merged segments.", len(merged_results))
        return merged_results

    def _align_by_segment(
        self,
        ts_segment: TranscriptSegment,
        speaker_segments: list[SpeakerSegment],
    ) -> AlignedSegment:
        """Find the speaker segment with the maximum temporal overlap."""
        max_overlap = 0.0
        best_speaker = "SPEAKER_UNKNOWN"

        for spk_segment in speaker_segments:
            # Calculate intersection
            overlap_start = max(ts_segment.start, spk_segment.start)
            overlap_end = min(ts_segment.end, spk_segment.end)
            overlap_duration = max(0.0, overlap_end - overlap_start)

            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                best_speaker = spk_segment.speaker

        # If no overlap found (e.g. ASR hallucinated in silence), assign to nearest or unknown
        if max_overlap == 0.0:
            best_speaker = self._find_nearest_speaker(
                ts_segment.start, ts_segment.end, speaker_segments
            )

        return AlignedSegment(
            id=ts_segment.id,
            speaker=best_speaker,
            text=ts_segment.text,
            start=ts_segment.start,
            end=ts_segment.end,
            language=ts_segment.language,
            confidence=ts_segment.confidence,
            words=ts_segment.words,
        )

    def _align_by_words(
        self,
        ts_segment: TranscriptSegment,
        speaker_segments: list[SpeakerSegment],
    ) -> AlignedSegment:
        """Assign speaker based on word-level overlap majority."""
        speaker_counts: dict[str, int] = {}
        speaker_durations: dict[str, float] = {}

        for word in ts_segment.words:
            word_speaker = "SPEAKER_UNKNOWN"
            max_overlap = 0.0
            
            # Find which speaker segment this word falls into
            for spk_segment in speaker_segments:
                overlap_start = max(word.start, spk_segment.start)
                overlap_end = min(word.end, spk_segment.end)
                overlap_duration = max(0.0, overlap_end - overlap_start)
                
                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    word_speaker = spk_segment.speaker

            if max_overlap == 0.0:
                word_speaker = self._find_nearest_speaker(
                    word.start, word.end, speaker_segments
                )

            speaker_counts[word_speaker] = speaker_counts.get(word_speaker, 0) + 1
            word_duration = word.end - word.start
            speaker_durations[word_speaker] = speaker_durations.get(word_speaker, 0.0) + word_duration

        # Choose the speaker with the most time spoken in this segment
        if speaker_durations:
            best_speaker = max(speaker_durations.items(), key=lambda x: x[1])[0]
        else:
            best_speaker = "SPEAKER_UNKNOWN"

        return AlignedSegment(
            id=ts_segment.id,
            speaker=best_speaker,
            text=ts_segment.text,
            start=ts_segment.start,
            end=ts_segment.end,
            language=ts_segment.language,
            confidence=ts_segment.confidence,
            words=ts_segment.words,
        )

    def _find_nearest_speaker(
        self,
        start: float,
        end: float,
        speaker_segments: list[SpeakerSegment],
    ) -> str:
        """Find the closest speaker temporally if there's no overlap."""
        midpoint = (start + end) / 2.0
        nearest_speaker = "SPEAKER_UNKNOWN"
        min_distance = float("inf")

        for spk_segment in speaker_segments:
            spk_midpoint = (spk_segment.start + spk_segment.end) / 2.0
            distance = abs(midpoint - spk_midpoint)
            if distance < min_distance:
                min_distance = distance
                nearest_speaker = spk_segment.speaker

        # If it's more than 5 seconds away, it's probably hallucinated noise
        if min_distance > 5.0:
            return "SPEAKER_UNKNOWN"
            
        return nearest_speaker

    def _merge_consecutive_segments(
        self,
        segments: list[AlignedSegment],
    ) -> list[AlignedSegment]:
        """Merge adjacent segments from the same speaker."""
        if not segments:
            return []

        merged = [segments[0]]
        
        for curr in segments[1:]:
            prev = merged[-1]
            
            # If same speaker, same language, and gap is less than 2 seconds
            if (
                curr.speaker == prev.speaker
                and curr.language == prev.language
                and (curr.start - prev.end) < 2.0
            ):
                # Calculate new confidence weighted by text length
                prev_len = len(prev.text)
                curr_len = len(curr.text)
                total_len = prev_len + curr_len
                
                new_conf = prev.confidence
                if total_len > 0:
                    new_conf = (
                        (prev.confidence * prev_len) + (curr.confidence * curr_len)
                    ) / total_len

                # Merge
                merged[-1] = AlignedSegment(
                    id=prev.id,  # Keep the first ID
                    speaker=prev.speaker,
                    text=f"{prev.text} {curr.text}",
                    start=prev.start,
                    end=curr.end,
                    language=prev.language,
                    confidence=new_conf,
                    words=prev.words + curr.words,
                )
            else:
                merged.append(curr)

        return merged
