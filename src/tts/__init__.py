"""
Text-to-Speech module with emotion and voice support.
"""

from .engine import (
    TTSEngine,
    Emotion,
    create_tts_engine,
    emotion_from_disposition,
)

__all__ = [
    "TTSEngine",
    "Emotion",
    "create_tts_engine",
    "emotion_from_disposition",
]

