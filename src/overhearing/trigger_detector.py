import logging
import re

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Set
from datetime import datetime

from src.rag.models import WorldLore

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Classification of player input."""
    IGNORE = "ignore"
    PLAYER_TO_PLAYER = "player_to_player"
    NPC_DIRECT = "npc_direct"
    NPC_INDIRECT = "npc_indirect"
    WORLD_ACTION = "world_action"


@dataclass
class TriggerResult:
    """Result of trigger detection."""
    trigger_type: TriggerType
    triggered_npc: Optional[str] = None
    confidence: float = 0.0
    reasoning: str = ""


class TriggerDetector:
    """
    Detects triggers in player speech to determine NPC activation.
    
    Uses rule-based pattern matching for speed and reliability.
    Can be enhanced with LLM classification for more complex scenarios.
    """
    
    # Patterns that indicate out-of-game conversation
    IGNORE_PATTERNS = [
        r"\b(pass|hand|give) me the (chips|drink|food|snacks|remote)\b",
        r"\b(can you|could you) (pause|stop|wait a sec)\b",
        r"\bbrb\b",
        r"\b(bathroom|toilet|restroom)\b",
        r"\b(real quick|one sec|hold on)\b",
    ]
    
    # Patterns for direct NPC address
    DIRECT_ADDRESS_PATTERNS = [
        r"\b(hey|hi|hello|greetings|excuse me|yo)[,\.]?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"\b(hey|hi|hello|greetings|excuse me|yo)[,\.]?\s+(\w+)",
        r"\bI\s+(ask|talk to|speak to|approach)\s+(?:the\s+)?(\w+)",
        r"\b(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)[,\s]+(can you|could you|do you|would you)",
        r"\b(?:the\s+)?(\w+)[,\s]+(can you|could you|do you|would you)",
        r",\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\??$",
        r",\s*(\w+)\s*\??$",
    ]
    
    # Patterns for explicit player-to-player communication (strategy, planning)
    STRATEGY_PATTERNS = [
        r"\b(what do|what should|should we)\b",
        r"\b(let's|let us)\b",
        r"\b(I think we|we should|we could)\b",
        r"\b(guys|everyone|team)\b",
        r"\b(attack him|use your spell|flank|cast|heal me)\b",
    ]
    
    # Patterns for narrative declarations (Movement, Time, Meta-game)
    NARRATIVE_PATTERNS = [
        r"\b(i|we|the party|the group|my character).{0,20}\s(go|goes|head|heads|walk|walks|travel|travels|leave|leaves|enter|enters)\s+(to|towards|into|out)\b",
        r"\b(i|we).{0,10}\s(want to|decide to|try to)\s+(go|leave|enter|head)\b",
        r"\b(next|following) (day|morning|night|evening|week)\b",
        r"\b(later|couple of hours) (that|later)\b",
        r"\b(fast forward|skip ahead|time passes)\b",
        r"\b(i|we)\s+(roll|cast|use|check).{0,10}\s(perception|stealth|initiative|spell|inventory|stats)\b",
        r"\b(that's|that is) (meta|out of character|ooc)\b",
        r"\bthe player\b",
    ]
    
    def __init__(self, world_lore: Optional[WorldLore] = None):
        """
        Initialize trigger detector.
        
        Args:
            world_lore: Optional world lore to extract NPC names and keywords
        """
        self.world_lore = world_lore
        self.npc_names: Set[str] = set()
        self.npc_keywords: dict[str, List[str]] = {}  # npc_id -> keywords
        
        self.last_npc_speaker: Optional[str] = None
        self.last_npc_time: Optional[datetime] = None
        self.active_window_duration: float = 20.0  # seconds
        
        if world_lore:
            self._extract_npc_info()
        
        logger.info(f"Trigger Detector initialized with {len(self.npc_names)} NPCs (Sticky Context Mode)")
    
    def _extract_npc_info(self):
        """Extract NPC names and associated keywords from world lore."""
        if not self.world_lore:
            return
        
        for npc in self.world_lore.npcs:
            self.npc_names.add(npc.name.lower())
            
            id_variations = npc.id.replace("_", " ").lower()
            self.npc_names.add(id_variations)
            
            keywords = []
            
            if npc.archetype:
                archetype_obj = self.world_lore.get_archetype(npc.archetype)
                if archetype_obj:
                    keywords.append(npc.archetype.replace("_", " "))
            
            if npc.location:
                keywords.append(npc.location.replace("_", " "))
            
            self.npc_keywords[npc.id] = keywords
    
    def add_npc(self, npc_id: str, name: str, keywords: Optional[List[str]] = None):
        """
        Args:
            npc_id: NPC identifier
            name: NPC display name
            keywords: Optional keywords associated with this NPC
        """
        self.npc_names.add(name.lower())
        self.npc_names.add(npc_id.replace("_", " ").lower())
        
        if keywords:
            self.npc_keywords[npc_id] = keywords
    
    def update_state_after_speech(self, npc_id: str):
        """
        Must be called after an NPC finishes speaking.
        Sets the 'Active Window' for this NPC : subsequent player speech
        (unless OOC) will be treated as a response.
        
        Args:
            npc_id: ID of the NPC who just finished speaking
        """
        self.last_npc_speaker = npc_id
        self.last_npc_time = datetime.now()
        logger.debug(f"Active Conversation Window set to: {npc_id} (will expire in {self.active_window_duration}s)")
    
    def reset_state(self):
        """
        Force reset of active conversation window.
        Useful when scene changes or context is explicitly cleared.
        """
        if self.last_npc_speaker:
            logger.debug(f"Resetting active conversation state (was: {self.last_npc_speaker})")
        self.last_npc_speaker = None
        self.last_npc_time = None
    
    def detect_trigger(
        self,
        transcription: str,
        active_npcs: Optional[List[str]] = None,
        boost_direct: bool = False,
        conversation_history: Optional[List] = None,
    ) -> TriggerResult:
        """
        Detect trigger in player transcription.
        
        Uses "Active Window" approach: if an NPC spoke recently (< 20s),
        all player speech (unless explicitly OOC) is treated as a response.
        
        Args:
            transcription: Player's transcribed speech
            active_npcs: List of NPCs currently in scene
            boost_direct: Whether to boost direct address confidence
            conversation_history: Optional conversation history for context
            
        Returns:
            TriggerResult with classification and NPC (if triggered)
        """
        text_lower = transcription.lower().strip()
        
        if len(text_lower) < 3:
            return TriggerResult(
                trigger_type=TriggerType.IGNORE,
                confidence=1.0,
                reasoning="Input too short",
            )
        
        for pattern in self.IGNORE_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return TriggerResult(
                    trigger_type=TriggerType.IGNORE,
                    confidence=0.9,
                    reasoning=f"Matched ignore pattern: {pattern}",
                )
        
        # FILTER: Narrative / Scene Change (cuts active context)
        if text_lower.startswith("the player") or " the player " in text_lower:
            self.last_npc_speaker = None
            self.last_npc_time = None
            logger.debug("Narrative statement detected (The Player), clearing active context")
            return TriggerResult(
                trigger_type=TriggerType.IGNORE,
                confidence=1.0,
                reasoning="Narrative (The Player detected)"
            )
        
        for pattern in self.NARRATIVE_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                self.last_npc_speaker = None
                self.last_npc_time = None
                logger.debug("Narrative statement detected, clearing active context")
                return TriggerResult(
                    trigger_type=TriggerType.IGNORE,
                    confidence=1.0,
                    reasoning="Narrative statement (scene change)"
                )
        
        # PRIORITY 1: Active Window
        if self.last_npc_speaker and self.last_npc_time:
            time_since = (datetime.now() - self.last_npc_time).total_seconds()
            
            if time_since < self.active_window_duration:
                is_strategy = any(re.search(p, text_lower, re.IGNORECASE) for p in self.STRATEGY_PATTERNS)
                
                if not is_strategy:
                    if active_npcs and self.last_npc_speaker in active_npcs:
                        logger.debug(f"Sticky Context: Responding to {self.last_npc_speaker} ({time_since:.1f}s ago)")
                        return TriggerResult(
                            trigger_type=TriggerType.NPC_DIRECT,
                            triggered_npc=self.last_npc_speaker,
                            confidence=0.95,
                            reasoning=f"Active Window: Continuing conversation with {self.last_npc_speaker} (spoke {time_since:.1f}s ago)"
                        )
                    else:
                        logger.debug(f"NPC {self.last_npc_speaker} no longer active in scene, clearing window")
                        self.last_npc_speaker = None
                        self.last_npc_time = None
            else:
                # Window expired, reset
                logger.debug(f"Active window expired ({time_since:.1f}s > {self.active_window_duration}s), resetting")
                self.last_npc_speaker = None
                self.last_npc_time = None
        
        # PRIORITY 2: Direct Address (Explicit NPC name mention)
        direct_result = self._check_direct_address(text_lower, active_npcs, boost_direct)
        if direct_result:
            return direct_result
        
        # PRIORITY 3: Indirect NPC mentions
        indirect_result = self._check_indirect_trigger(text_lower, active_npcs)
        if indirect_result:
            return indirect_result
        
        # PRIORITY 4: Player-to-player strategy patterns
        for pattern in self.STRATEGY_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return TriggerResult(
                    trigger_type=TriggerType.PLAYER_TO_PLAYER,
                    confidence=0.8,
                    reasoning=f"Matched strategy pattern: {pattern}",
                )
        
        world_result = self._check_world_action(text_lower, active_npcs)
        if world_result:
            return world_result
        
        return TriggerResult(
            trigger_type=TriggerType.PLAYER_TO_PLAYER,
            confidence=0.5,
            reasoning="No clear trigger detected, assuming player conversation",
        )
    
    def _check_direct_address(
        self,
        text_lower: str,
        active_npcs: Optional[List[str]],
        boost_direct: bool = False,
    ) -> Optional[TriggerResult]:
        """Check for direct address to an NPC."""
        for npc_name in self.npc_names:
            npc_name_lower = npc_name.lower()
            if npc_name_lower in text_lower:
                npc_id = self._find_npc_id_by_name(npc_name)
                
                if active_npcs and npc_id in active_npcs:
                    if (text_lower.endswith(npc_name_lower) or 
                        text_lower.endswith(npc_name_lower + "?") or
                        f", {npc_name_lower}" in text_lower or
                        f", {npc_name_lower}?" in text_lower):
                        confidence = 1.0
                    else:
                        confidence = 0.9
                    
                    if boost_direct:
                        confidence = 1.0
                    
                    return TriggerResult(
                        trigger_type=TriggerType.NPC_DIRECT,
                        triggered_npc=npc_id,
                        confidence=confidence,
                        reasoning=f"Direct address detected: '{npc_name}' mentioned",
                    )
        
        for pattern in self.DIRECT_ADDRESS_PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                potential_name = match.group(match.lastindex).lower()
                
                for npc_name in self.npc_names:
                    if potential_name in npc_name or npc_name in potential_name:
                        npc_id = self._find_npc_id_by_name(npc_name)
                        
                        confidence = 0.95
                        if (boost_direct or active_npcs) and npc_id in active_npcs:
                            confidence = 1.0
                        
                        return TriggerResult(
                            trigger_type=TriggerType.NPC_DIRECT,
                            triggered_npc=npc_id,
                            confidence=confidence,
                            reasoning=f"Direct address detected: '{potential_name}'",
                        )
        
        return None
    
    def _check_indirect_trigger(
        self,
        text_lower: str,
        active_npcs: Optional[List[str]],
    ) -> Optional[TriggerResult]:
        for npc_name in self.npc_names:
            if npc_name in text_lower:
                npc_id = self._find_npc_id_by_name(npc_name)
                
                confidence = 0.7
                if active_npcs and npc_id in active_npcs:
                    confidence = 0.85
                
                return TriggerResult(
                    trigger_type=TriggerType.NPC_INDIRECT,
                    triggered_npc=npc_id,
                    confidence=confidence,
                    reasoning=f"NPC name mentioned: '{npc_name}'",
                )
        
        if active_npcs:
            for npc_id in active_npcs:
                if npc_id in self.npc_keywords:
                    for keyword in self.npc_keywords[npc_id]:
                        if keyword.lower() in text_lower:
                            return TriggerResult(
                                trigger_type=TriggerType.NPC_INDIRECT,
                                triggered_npc=npc_id,
                                confidence=0.6,
                                reasoning=f"Keyword matched for {npc_id}: '{keyword}'",
                            )
        
        return None
    
    def _check_world_action(
        self,
        text_lower: str,
        active_npcs: Optional[List[str]],
    ) -> Optional[TriggerResult]:
        action_patterns = {
            "enter": r"\b(enter|go into|walk into|step into)\b",
            "look": r"\b(look at|examine|inspect|observe)\b",
            "take": r"\b(take|grab|pick up|steal)\b",
            "attack": r"\b(attack|fight|hit|strike)\b",
        }
        
        for action_type, pattern in action_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                if active_npcs:
                    return TriggerResult(
                        trigger_type=TriggerType.WORLD_ACTION,
                        triggered_npc=active_npcs[0],
                        confidence=0.5,
                        reasoning=f"World action detected: {action_type}",
                    )
        
        return None
    
    def _find_npc_id_by_name(self, npc_name: str) -> Optional[str]:
        if not self.world_lore:
            return npc_name

        for npc in self.world_lore.npcs:
            if npc.name.lower() == npc_name or npc.id.replace("_", " ").lower() == npc_name:
                return npc.id
        
        return npc_name
