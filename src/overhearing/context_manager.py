import logging
import threading
import re
import requests

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

from src.rag.models import SessionState, WorldLore

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    speaker: str  # "player", "npc", "system"
    text: str
    timestamp: datetime
    npc_id: Optional[str] = None


@dataclass
class GameContext:
    """
    Holds data about the location, time and events.
    """
    current_location: str = "unknown"
    current_location_id: Optional[str] = None
    time_of_day: str = "day"
    scene_mood: str = "neutral"

    # Active entities
    active_npcs: List[str] = field(default_factory=list)  # NPCs in current scene
    active_quests: List[str] = field(default_factory=list)
    world_events: List[str] = field(default_factory=list)
    
    conversation_history: deque = field(default_factory=lambda: deque(maxlen=20))
    last_update: datetime = field(default_factory=datetime.now)
    
    def to_session_state(self) -> SessionState:
        """
        Returns:
            SessionState for RAG module
        """
        return SessionState(
            current_time=self.time_of_day,
            current_location=self.current_location,
            current_mood=self.scene_mood,
            active_quests=self.active_quests,
            world_events=self.world_events,
        )


class ContextManager:
    """
    Manages game context and conversation history.
    
    Thread-safe that tracks:
    - Current location, time, and mood
    - Active NPCs in the scene
    - Conversation history
    """
    
    def __init__(
        self,
        initial_session: Optional[SessionState] = None,
        world_lore: Optional[WorldLore] = None,
        max_history: int = 20,
    ):
        """
        Args:
            initial_session: Optional initial session state
            world_lore: Optional world lore for location detection
            max_history: Maximum conversation turns to keep in history
        """
        self.context = GameContext()
        self._lock = threading.RLock()
        self.world_lore = world_lore
        
        if initial_session:
            self.load_session(initial_session)
        
    def load_session(self, session: SessionState):
        """
        Load session state into context.
        
        Args:
            session: Session state from YAML
        """
        with self._lock:
            self.context.current_location = session.current_location
            self.context.time_of_day = session.current_time
            self.context.scene_mood = session.current_mood
            self.context.active_quests = session.active_quests.copy()
            self.context.world_events = session.world_events.copy()
            self.context.last_update = datetime.now()
        
        logger.info(f"Loaded session: {session.current_location} at {session.current_time}")
    
    def update_location(self, location: str, location_id: Optional[str] = None):
        """
        Args:
            location: Human-readable location name
            location_id: Optional location ID from world lore
        """
        with self._lock:
            self.context.current_location = location
            self.context.current_location_id = location_id
            self.context.last_update = datetime.now()
        
        logger.info(f"Location updated: {location}")

        requests.post("http://localhost:5000/scene",
            json={"scene": location}
        )
    
    def update_time(self, time_of_day: str):
        """
        Args:
            time_of_day: morning, day, evening, or night
        """
        with self._lock:
            self.context.time_of_day = time_of_day
            self.context.last_update = datetime.now()
        
        logger.info(f"Time updated: {time_of_day}")
    
    def update_mood(self, mood: str):
        """
        Args:
            mood: Scene mood (tense, relaxed, mysterious, etc.)
        """
        with self._lock:
            self.context.scene_mood = mood
            self.context.last_update = datetime.now()
        
        logger.info(f"Mood updated: {mood}")
    
    def add_npc_to_scene(self, npc_id: str):
        """
        Args:
            npc_id: NPC identifier
        """
        with self._lock:
            if npc_id not in self.context.active_npcs:
                self.context.active_npcs.append(npc_id)
        
        logger.debug(f"NPC added to scene: {npc_id}")
    
    def detect_context_changes(self, transcription: str) -> Dict[str, Any]:
        """
        Detect context changes from player speech with smarter parsing.
        Handles 'leave A to go to B' logic.
        """
        changes = {}
        text_lower = transcription.lower()
        location_patterns = {}
        found_locations = []
        
        if self.world_lore and self.world_lore.locations:
            for location in self.world_lore.locations:
                patterns = []
                if location.name:
                    patterns.append(location.name.lower())
                    patterns.append(location.name.lower().replace("the ", "").strip())
                
                if location.id:
                    patterns.append(location.id.replace("_", " "))
                
                if location.name:
                    words = location.name.lower().split()
                    key_words = [w for w in words if w not in ["the", "of", "a", "an", "and", "or", "forgotten", "tower"]]
                    patterns.extend(key_words)
                
                if patterns:
                    location_patterns[location.id] = list(set(patterns))

        for loc_id, patterns in location_patterns.items():
            for pattern in patterns:
                try:
                    for match in re.finditer(r"\b" + re.escape(pattern) + r"\b", text_lower):
                        found_locations.append((match.start(), loc_id))
                except Exception:
                    continue

        if found_locations:
            found_locations.sort(key=lambda x: x[0])
            
            chosen_loc = None
            
            for index, loc_id in found_locations:
                start_window = max(0, index - 25)
                context_window = text_lower[start_window:index]
                
                is_leaving = any(word in context_window for word in ["leave", "leaving", "exit", "out of", "from", "bye"])
                
                if is_leaving:
                    logger.info(f"Ignoring location '{loc_id}' because context implies departure: '{context_window}'")
                    continue
                
                chosen_loc = loc_id
            
            if chosen_loc:
                changes["location"] = chosen_loc

        time_patterns = {
            "morning": ["morning", "dawn", "sunrise"],
            "day": ["day", "afternoon", "noon"],
            "evening": ["evening", "dusk", "sunset"],
            "night": ["night", "midnight", "dark"],
        }
        for time_id, patterns in time_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                changes["time"] = time_id
        
        if any(word in text_lower for word in ["danger", "alert", "attack", "fight", "blood"]):
            changes["mood"] = "tense"
        elif any(word in text_lower for word in ["mysterious", "strange", "odd", "fog", "ghost"]):
            changes["mood"] = "mysterious"
        elif any(word in text_lower for word in ["calm", "peaceful", "relaxed", "quiet"]):
            changes["mood"] = "relaxed"
        
        return changes
    
    def add_conversation_turn(
        self,
        speaker: str,
        text: str,
        npc_id: Optional[str] = None,
    ):
        turn = ConversationTurn(
            speaker=speaker,
            text=text,
            timestamp=datetime.now(),
            npc_id=npc_id,
        )
        
        with self._lock:
            self.context.conversation_history.append(turn)
    
    def get_recent_history(self, n: int = 10) -> List[ConversationTurn]:
        with self._lock:
            history = list(self.context.conversation_history)
            return history[-n:] if len(history) > n else history
    
    def get_context_snapshot(self) -> GameContext:
        """
        Returns:
            Copy of current GameContext
        """
        with self._lock:
            return GameContext(
                current_location=self.context.current_location,
                current_location_id=self.context.current_location_id,
                time_of_day=self.context.time_of_day,
                scene_mood=self.context.scene_mood,
                active_npcs=self.context.active_npcs.copy(),
                active_quests=self.context.active_quests.copy(),
                world_events=self.context.world_events.copy(),
                conversation_history=self.context.conversation_history.copy(),
                last_update=self.context.last_update,
            )
    
    def get_context_summary(self) -> str:
        """
        Returns:
            Context summary string
        """
        with self._lock:
            lines = [
                f"Location: {self.context.current_location}",
                f"Time: {self.context.time_of_day}",
                f"Mood: {self.context.scene_mood}",
            ]
            
            if self.context.active_npcs:
                lines.append(f"NPCs present: {', '.join(self.context.active_npcs)}")
            
            if self.context.active_quests:
                lines.append(f"Active quests: {len(self.context.active_quests)}")
            
            history_count = len(self.context.conversation_history)
            if history_count > 0:
                lines.append(f"Recent conversation: {history_count} turns")
            
            return "\n".join(lines)
    
    def apply_detected_changes(self, changes: Dict[str, Any]):
        if "location" in changes:
            self.update_location(changes["location"])
        
        if "time" in changes:
            self.update_time(changes["time"])
        
        if "mood" in changes:
            self.update_mood(changes["mood"])

