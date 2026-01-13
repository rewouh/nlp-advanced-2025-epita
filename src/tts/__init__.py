"""
Text-to-Speech module with emotion and voice support.
"""

from .engine import (
    TTSEngine,
    Emotion,
    emotion_from_disposition,
)

__all__ = [
    "TTSEngine",
    "Emotion",
    "emotion_from_disposition",
]

