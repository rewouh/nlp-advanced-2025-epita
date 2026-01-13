"""
Speech-to-Text module with faster-whisper and RPG hotwords support.
"""

from .engine import STTEngine, HotwordExtractor, create_stt_engine

__all__ = ["STTEngine", "HotwordExtractor", "create_stt_engine"]

