"""
Overhearing module - Context tracking and trigger detection.

This module continuously monitors conversations and determines when
NPCs should activate and respond.
"""

from .context_manager import ContextManager, GameContext
from .trigger_detector import TriggerDetector, TriggerResult, TriggerType

__all__ = [
    "ContextManager",
    "GameContext",
    "TriggerDetector",
    "TriggerResult",
    "TriggerType",
]

