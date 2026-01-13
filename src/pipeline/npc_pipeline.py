"""
NPC Pipeline - Handles NPC activation and response generation.

Integrates RAG, Orchestrator, and TTS for complete NPC interactions.
"""

import logging
from typing import Optional, Dict

from src.conversation import (
    NPCOrchestrator,
    Persona,
    GlobalState,
    create_npc,
)
from src.rag.lore_manager import LoreManager
from src.rag.models import WorldLore, NPCDefinition

logger = logging.getLogger(__name__)


class NPCPipeline:
    """
    Manages NPC activation and response generation.
    
    Handles the complete flow from NPC trigger to TTS output:
    1. Get NPC definition from RAG
    2. Retrieve relevant lore
    3. Create/get orchestrator
    4. Generate response
    5. Return response for TTS
    """
    
    def __init__(
        self,
        lore_manager: LoreManager,
        model: str = "qwen2.5:3b",
        temperature: float = 0.7,
    ):
        """
        Initialize NPC pipeline.
        
        Args:
            lore_manager: LoreManager instance with loaded world data
            model: LLM model name for Ollama
            temperature: Generation temperature
        """
        self.lore_manager = lore_manager
        self.model = model
        self.temperature = temperature
        
        # Active NPC orchestrators (npc_id -> orchestrator)
        self.active_npcs: Dict[str, NPCOrchestrator] = {}
        
        logger.info(f"NPC Pipeline initialized with model: {model}")
    
    def activate_npc(
        self,
        npc_id: str,
        player_input: str,
        current_location: Optional[str] = None,
        time_of_day: str = "day",
        scene_mood: str = "neutral",
    ) -> str:
        """
        Activate NPC and generate response.
        
        Args:
            npc_id: NPC identifier
            player_input: What the player said
            current_location: Current location
            time_of_day: Time of day
            scene_mood: Scene mood
            
        Returns:
            NPC's response text (for TTS)
        """
        logger.info(f"Activating NPC: {npc_id}")
        
        # Get or create orchestrator
        orchestrator = self._get_or_create_orchestrator(npc_id)
        
        # Update global state
        global_state = GlobalState(
            location=current_location or "unknown",
            time_of_day=time_of_day,
            scene_mood=scene_mood,
        )
        orchestrator.update_state(global_state)
        
        # Retrieve relevant lore
        lore_context = self.lore_manager.build_context_for_npc(
            query=player_input,
            npc_id=npc_id,
            n_results=3,
        )
        orchestrator.set_lore(lore_context)
        
        # Generate response
        response = orchestrator.think_sync(player_input)
        
        logger.info(f"NPC {npc_id} responded: {response[:50]}...")
        
        return response
    
    def activate_npc_streaming(
        self,
        npc_id: str,
        player_input: str,
        current_location: Optional[str] = None,
        time_of_day: str = "day",
        scene_mood: str = "neutral",
    ):
        """
        Activate NPC and generate response with streaming.
        
        Yields tokens as they're generated.
        
        Args:
            npc_id: NPC identifier
            player_input: What the player said
            current_location: Current location
            time_of_day: Time of day
            scene_mood: Scene mood
            
        Yields:
            Response tokens
        """
        logger.info(f"Activating NPC (streaming): {npc_id}")
        
        # Get or create orchestrator
        orchestrator = self._get_or_create_orchestrator(npc_id)
        
        # Update global state
        global_state = GlobalState(
            location=current_location or "unknown",
            time_of_day=time_of_day,
            scene_mood=scene_mood,
        )
        orchestrator.update_state(global_state)
        
        # Retrieve relevant lore
        lore_context = self.lore_manager.build_context_for_npc(
            query=player_input,
            npc_id=npc_id,
            n_results=3,
        )
        orchestrator.set_lore(lore_context)
        
        # Generate response with streaming
        for token in orchestrator.think(player_input):
            yield token
    
    def _get_or_create_orchestrator(self, npc_id: str) -> NPCOrchestrator:
        """
        Get existing orchestrator or create new one.
        
        Args:
            npc_id: NPC identifier
            
        Returns:
            NPCOrchestrator instance
        """
        if npc_id in self.active_npcs:
            return self.active_npcs[npc_id]
        
        # Get NPC definition from lore
        npc_def = self.lore_manager.get_npc_definition(npc_id)
        
        if npc_def:
            # Create from definition
            orchestrator = self._create_from_definition(npc_def)
        else:
            # Try to create from archetype
            orchestrator = self._create_from_archetype(npc_id)
        
        # Cache orchestrator
        self.active_npcs[npc_id] = orchestrator
        
        return orchestrator
    
    def _create_from_definition(self, npc_def: NPCDefinition) -> NPCOrchestrator:
        """Create orchestrator from NPC definition."""
        # Get base archetype traits if archetype specified
        base_traits = []
        motives = ""
        
        if npc_def.archetype and self.lore_manager.world_data:
            archetype = self.lore_manager.world_data.get_archetype(npc_def.archetype)
            if archetype:
                base_traits = archetype.traits.copy()
                motives = archetype.motives
        
        # Merge with custom traits
        all_traits = list(set(base_traits + npc_def.traits))
        
        # Create orchestrator
        return create_npc(
            npc_id=npc_def.id,
            traits=all_traits,
            motives=motives,
            private_knowledge=npc_def.secrets,
            model=self.model,
        )
    
    def _create_from_archetype(self, npc_id: str) -> NPCOrchestrator:
        """Create orchestrator from archetype (fallback)."""
        # Try to extract archetype from npc_id
        # e.g., "merchant_1" -> "merchant"
        potential_archetype = npc_id.split("_")[0]
        
        if self.lore_manager.world_data:
            archetype = self.lore_manager.world_data.get_archetype(potential_archetype)
            if archetype:
                logger.info(f"Creating NPC from archetype: {potential_archetype}")
                return create_npc(
                    npc_id=npc_id,
                    traits=archetype.traits,
                    motives=archetype.motives,
                    model=self.model,
                )
        
        # Ultimate fallback: generic NPC
        logger.warning(f"Creating generic NPC: {npc_id}")
        return create_npc(
            npc_id=npc_id,
            traits=["neutral", "helpful"],
            motives="Assist travelers",
            model=self.model,
        )
    
    def interrupt_npc(self, npc_id: str):
        """
        Interrupt NPC generation.
        
        Args:
            npc_id: NPC identifier
        """
        if npc_id in self.active_npcs:
            self.active_npcs[npc_id].interrupt()
            logger.info(f"Interrupted NPC: {npc_id}")
    
    def reset_npc_memory(self, npc_id: str):
        """
        Reset NPC conversation memory.
        
        Args:
            npc_id: NPC identifier
        """
        if npc_id in self.active_npcs:
            self.active_npcs[npc_id].reset_memory()
            logger.info(f"Reset memory for NPC: {npc_id}")
    
    def deactivate_npc(self, npc_id: str):
        """
        Deactivate NPC (remove from active list).
        
        Args:
            npc_id: NPC identifier
        """
        if npc_id in self.active_npcs:
            del self.active_npcs[npc_id]
            logger.info(f"Deactivated NPC: {npc_id}")
    
    def get_active_npc_ids(self) -> list[str]:
        """
        Get list of active NPC IDs.
        
        Returns:
            List of NPC IDs with active orchestrators
        """
        return list(self.active_npcs.keys())

