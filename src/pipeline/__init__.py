"""
Pipeline module - Integrates all components into a complete system.
"""

from .npc_pipeline import NPCPipeline
from .overhearing_pipeline import OverhearingPipeline

__all__ = ["NPCPipeline", "OverhearingPipeline"]

