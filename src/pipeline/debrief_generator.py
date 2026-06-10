"""
Meeting Intelligence System — Debrief Generator
===============================================
Generates structured meeting debriefs from transcribed and enriched segments
using FLAN-T5 (instruction-tuned LLM).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional, Any

from config.settings import settings
from src.models.schemas import (
    ActionItem,
    Conflict,
    Decision,
    EnrichedSegment,
    MeetingDebrief,
    SpeakerProfile,
)
from src.utils.device_manager import device_manager

logger = logging.getLogger(__name__)


class DebriefGenerator:
    """Wrapper for FLAN-T5 to generate structured debriefs."""

    def __init__(self) -> None:
        self.config = settings.debrief
        self.device = device_manager.get_device(self.config.device)
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None

    def load_model(self) -> None:
        """Load FLAN-T5 model and tokenizer into memory."""
        if self.model is not None and self.tokenizer is not None:
            return

        logger.info(
            "Loading Debrief Generator %s (Device: %s)",
            self.config.model_name,
            self.device.type,
        )

        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer

            self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)
            
            # Use float16 for large models to save VRAM if on GPU
            torch_dtype = "auto"
            if self.device.type == "cuda" and "large" in self.config.model_name or "xl" in self.config.model_name:
                import torch
                torch_dtype = torch.float16

            self.model = T5ForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype=torch_dtype,
            )
            self.model.to(self.device)
            
            logger.info("Debrief Generator loaded successfully")

        except Exception as e:
            logger.error("Failed to load Debrief Generator: %s", e)
            raise RuntimeError(f"Debrief Generator initialization failed: {e}") from e

    def generate_debrief(self, segments: list[EnrichedSegment]) -> MeetingDebrief:
        """
        Generate a complete meeting debrief from segments.

        Parameters
        ----------
        segments : list[EnrichedSegment]
            List of processed segments containing text, speaker, emotion, intent.

        Returns
        -------
        MeetingDebrief
            Structured summary of the meeting.
        """
        if not segments:
            logger.warning("Empty segments provided to debrief generator")
            return MeetingDebrief(summary="Empty meeting transcript.")

        self.load_model()
        
        # 1. Prepare transcript string
        transcript_text = self._format_transcript(segments)
        
        # If the transcript is very long, we should chunk it.
        # For simplicity in this implementation, we will process it in one go,
        # but in production, we would use LangChain or similar for map-reduce summarization.
        # We will truncate for now to fit in max_input_tokens if extremely long.
        
        # 2. Extract components
        summary = self._generate_summary(transcript_text)
        decisions = self._extract_decisions(transcript_text)
        action_items = self._extract_action_items(transcript_text)
        conflicts = self._extract_conflicts(transcript_text)
        
        # 3. Compute speaker profiles
        speaker_profiles = self._compute_speaker_profiles(segments)

        return MeetingDebrief(
            summary=summary,
            decisions=decisions,
            action_items=action_items,
            conflicts=conflicts,
            speaker_profiles=speaker_profiles,
        )

    def _format_transcript(self, segments: list[EnrichedSegment]) -> str:
        """Format segments into a readable transcript."""
        lines = []
        for seg in segments:
            # e.g., "SPEAKER_00: Hello everyone."
            lines.append(f"{seg.speaker}: {seg.text}")
        return "\n".join(lines)

    def _generate(self, prompt: str) -> str:
        """Helper to run text generation."""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=self.config.max_input_tokens, 
            truncation=True
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
            num_beams=self.config.num_beams,
            do_sample=self.config.temperature > 0,
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _generate_summary(self, transcript: str) -> str:
        """Generate a general summary."""
        prompt = f"Summarize the following meeting transcript in 3-4 sentences. Focus on the main topics discussed.\n\nTranscript:\n{transcript}\n\nSummary:"
        return self._generate(prompt)

    def _extract_decisions(self, transcript: str) -> list[Decision]:
        """Extract decisions made during the meeting."""
        prompt = (
            "Extract the key decisions made in the following meeting transcript. "
            "Format each decision as a bullet point starting with a hyphen (-).\n\n"
            f"Transcript:\n{transcript}\n\nDecisions:"
        )
        
        response = self._generate(prompt)
        
        # Parse bullet points
        decisions = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('*'):
                desc = line[1:].strip()
                if desc:
                    decisions.append(Decision(description=desc))
                    
        return decisions

    def _extract_action_items(self, transcript: str) -> list[ActionItem]:
        """Extract action items and assignees."""
        prompt = (
            "Extract the action items assigned in the following meeting transcript. "
            "For each action item, identify the owner if possible. "
            "Format each action item as '- [Owner]: [Task]'. If no owner, just '- [Task]'.\n\n"
            f"Transcript:\n{transcript}\n\nAction Items:"
        )
        
        response = self._generate(prompt)
        
        action_items = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('*'):
                content = line[1:].strip()
                if not content:
                    continue
                
                # Try to parse "[Owner]: [Task]"
                match = re.match(r'^\[?([^\]:]+)\]?:\s*(.*)', content)
                if match:
                    owner, task = match.groups()
                    action_items.append(ActionItem(description=task.strip(), owner=owner.strip()))
                else:
                    action_items.append(ActionItem(description=content))
                    
        return action_items

    def _extract_conflicts(self, transcript: str) -> list[Conflict]:
        """Extract conflicts or disagreements."""
        prompt = (
            "Identify any disagreements, conflicts, or unresolved issues in the following meeting transcript. "
            "Format each conflict as a bullet point starting with a hyphen (-). "
            "If there are none, reply with 'None'.\n\n"
            f"Transcript:\n{transcript}\n\nConflicts:"
        )
        
        response = self._generate(prompt)
        
        if response.strip().lower() in ('none', 'none.'):
            return []
            
        conflicts = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('*'):
                desc = line[1:].strip()
                if desc:
                    conflicts.append(Conflict(description=desc))
                    
        return conflicts

    def _compute_speaker_profiles(self, segments: list[EnrichedSegment]) -> list[SpeakerProfile]:
        """Compute profiles based on emotion and intent aggregates."""
        profiles_map = {}
        
        for seg in segments:
            spk = seg.speaker
            if spk not in profiles_map:
                profiles_map[spk] = {
                    "duration": 0.0,
                    "turns": 0,
                    "emotions": {},
                    "intents": {}
                }
                
            p = profiles_map[spk]
            duration = seg.end - seg.start
            p["duration"] += duration
            p["turns"] += 1
            
            emo = seg.emotion.label
            p["emotions"][emo] = p["emotions"].get(emo, 0) + duration
            
            intent = seg.intent.label
            p["intents"][intent] = p["intents"].get(intent, 0) + 1

        results = []
        for spk, p in profiles_map.items():
            dominant_emo = max(p["emotions"].items(), key=lambda x: x[1])[0] if p["emotions"] else "neutral"
            
            # Simple heuristic for communication style based on intents
            style = "analytical"
            if p["intents"].get("action_request", 0) > p["turns"] * 0.2:
                style = "assertive"
            elif p["intents"].get("agreement", 0) > p["turns"] * 0.2:
                style = "collaborative"
            
            results.append(
                SpeakerProfile(
                    speaker=spk,
                    total_speaking_time_seconds=round(p["duration"], 1),
                    turn_count=p["turns"],
                    dominant_emotion=dominant_emo,
                    communication_style=style,
                )
            )
            
        return results

    def unload_model(self) -> None:
        """Free GPU memory if needed."""
        self.model = None
        self.tokenizer = None
        if self.device.type == "cuda":
            import torch
            torch.cuda.empty_cache()
            logger.info("Debrief Generator unloaded from VRAM")
