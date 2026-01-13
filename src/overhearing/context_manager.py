"""
Context Manager - Tracks current game state and conversation history.

This module maintains the dynamic state of the game world including:
- Current location and location ID
- Time of day (morning, day, evening, night)
- Scene mood (tense, relaxed, mysterious, etc.)
- Active NPCs in the current scene
- Active quests and world events
- Recent conversation history (last N turns)

The ContextManager is thread-safe and provides methods to:
- Update location, time, and mood
- Add/remove NPCs from the scene
- Detect context changes from player speech
- Track conversation history
- Generate context snapshots for NPC activation

Context changes are detected from player transcriptions using pattern
matching. The system dynamically extracts location patterns from world
lore to avoid hardcoding, and implements "Last Valid Destination" logic
to handle phrases like "leave A to go to B".

Usage:
    manager = ContextManager(initial_session=session, world_lore=world)
    changes = manager.detect_context_changes("I go to the tavern")
    if "location" in changes:
        manager.apply_detected_changes(changes)
    context = manager.get_context_snapshot()
"""

import logging
import threading
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

from src.rag.models import SessionState, WorldLore

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    speaker: str  # "player", "npc", "system"
    text: str
    timestamp: datetime
    npc_id: Optional[str] = None  # If speaker is NPC, which NPC


@dataclass
class GameContext:
    """
    Current game context state.
    
    This is the dynamic state that gets updated as players interact
    with the world and NPCs.
    """
    # Location and time
    current_location: str = "unknown"
    current_location_id: Optional[str] = None
    time_of_day: str = "day"  # morning, day, evening, night
    scene_mood: str = "neutral"  # tense, relaxed, mysterious, etc.
    
    # Active entities
    active_npcs: List[str] = field(default_factory=list)  # NPCs in current scene
    active_quests: List[str] = field(default_factory=list)
    world_events: List[str] = field(default_factory=list)
    
    # Conversation history (last N turns)
    conversation_history: deque = field(default_factory=lambda: deque(maxlen=20))
    
    # Metadata
    last_update: datetime = field(default_factory=datetime.now)
    
    def to_session_state(self) -> SessionState:
        """
        Convert to RAG SessionState format.
        
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
    
    Thread-safe manager that tracks:
    - Current location, time, and mood
    - Active NPCs in the scene
    - Recent conversation history
    - Context updates from transcriptions
    """
    
    def __init__(
        self,
        initial_session: Optional[SessionState] = None,
        world_lore: Optional[WorldLore] = None,
        max_history: int = 20,
    ):
        """
        Initialize context manager.
        
        Args:
            initial_session: Optional initial session state
            world_lore: Optional world lore for location detection
            max_history: Maximum conversation turns to keep in history
        """
        self.context = GameContext()
        self._lock = threading.RLock()
        self.world_lore = world_lore
        
        # Load initial session if provided
        if initial_session:
            self.load_session(initial_session)
        
        logger.info("Context Manager initialized")
    
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
        Update current location.
        
        Args:
            location: Human-readable location name
            location_id: Optional location ID from world lore
        """
        with self._lock:
            self.context.current_location = location
            self.context.current_location_id = location_id
            self.context.last_update = datetime.now()
        
        logger.info(f"Location updated: {location}")
    
    def update_time(self, time_of_day: str):
        """
        Update time of day.
        
        Args:
            time_of_day: morning, day, evening, or night
        """
        with self._lock:
            self.context.time_of_day = time_of_day
            self.context.last_update = datetime.now()
        
        logger.info(f"Time updated: {time_of_day}")
    
    def update_mood(self, mood: str):
        """
        Update scene mood.
        
        Args:
            mood: Scene mood (tense, relaxed, mysterious, etc.)
        """
        with self._lock:
            self.context.scene_mood = mood
            self.context.last_update = datetime.now()
        
        logger.info(f"Mood updated: {mood}")
    
    def add_npc_to_scene(self, npc_id: str):
        """
        Add NPC to current scene.
        
        Args:
            npc_id: NPC identifier
        """
        with self._lock:
            if npc_id not in self.context.active_npcs:
                self.context.active_npcs.append(npc_id)
        
        logger.debug(f"NPC added to scene: {npc_id}")
    
    def remove_npc_from_scene(self, npc_id: str):
        """
        Remove NPC from current scene.
        
        Args:
            npc_id: NPC identifier
        """
        with self._lock:
            if npc_id in self.context.active_npcs:
                self.context.active_npcs.remove(npc_id)
        
        logger.debug(f"NPC removed from scene: {npc_id}")
    
    def detect_context_changes(self, transcription: str) -> Dict[str, Any]:
        """
        Detect context changes from player speech with smarter parsing.
        Handles 'leave A to go to B' logic.
        """
        changes = {}
        text_lower = transcription.lower()
        
        # Build location patterns from world lore
        location_patterns = {}
        
        if self.world_lore and self.world_lore.locations:
            for location in self.world_lore.locations:
                patterns = []
                # Full name (e.g., "The Rusty Anchor Tavern")
                if location.name:
                    patterns.append(location.name.lower())
                    patterns.append(location.name.lower().replace("the ", "").strip())
                
                # ID variations (e.g., "rusty_anchor")
                if location.id:
                    patterns.append(location.id.replace("_", " "))
                
                # Key words from name (e.g., "Tavern", "Anchor")
                if location.name:
                    words = location.name.lower().split()
                    key_words = [w for w in words if w not in ["the", "of", "a", "an", "and", "or", "forgotten", "tower"]]
                    patterns.extend(key_words)
                
                if patterns:
                    location_patterns[location.id] = list(set(patterns))
        else:
            # Fallback
            location_patterns = {
                "rusty_anchor": ["tavern", "bar", "inn", "rusty anchor"],
                "frozen_harbor": ["harbor", "port", "docks", "pier", "dock"],
                "scholars_tower": ["tower", "library"],
                "guard_barracks": ["barracks", "guard post", "keep"],
            }
        
        # Search for all locations in the phrase
        found_locations = []  # List of tuples (index, location_id)
        
        for loc_id, patterns in location_patterns.items():
            for pattern in patterns:
                # Use regex to avoid false positives (e.g., "port" in "important")
                try:
                    for match in re.finditer(r"\b" + re.escape(pattern) + r"\b", text_lower):
                        found_locations.append((match.start(), loc_id))
                except Exception:
                    continue

        # Analyze context (departure vs arrival)
        if found_locations:
            # Sort by appearance order in the phrase
            found_locations.sort(key=lambda x: x[0])
            
            chosen_loc = None
            
            for index, loc_id in found_locations:
                # Look at 25 characters BEFORE the found word
                start_window = max(0, index - 25)
                context_window = text_lower[start_window:index]
                
                # Keywords indicating departure (leaving this location)
                is_leaving = any(word in context_window for word in ["leave", "leaving", "exit", "out of", "from", "bye"])
                
                if is_leaving:
                    logger.info(f"Ignoring location '{loc_id}' because context implies departure: '{context_window}'")
                    continue
                
                # If not a departure, it's a potential destination
                # As we traverse left to right, the last validated location
                # will be the final destination ("I pass through A to go to B")
                chosen_loc = loc_id
            
            # If we found a valid destination
            if chosen_loc:
                changes["location"] = chosen_loc

        # Time detection
        time_patterns = {
            "morning": ["morning", "dawn", "sunrise"],
            "day": ["day", "afternoon", "noon"],
            "evening": ["evening", "dusk", "sunset"],
            "night": ["night", "midnight", "dark"],
        }
        for time_id, patterns in time_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                changes["time"] = time_id
        
        # Mood detection
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
        """
        Add a conversation turn to history.
        
        Args:
            speaker: Speaker type (player, npc, system)
            text: What was said
            npc_id: If NPC, which NPC
        """
        turn = ConversationTurn(
            speaker=speaker,
            text=text,
            timestamp=datetime.now(),
            npc_id=npc_id,
        )
        
        with self._lock:
            self.context.conversation_history.append(turn)
    
    def get_recent_history(self, n: int = 10) -> List[ConversationTurn]:
        """
        Get recent conversation history.
        
        Args:
            n: Number of recent turns to get
            
        Returns:
            List of recent conversation turns
        """
        with self._lock:
            history = list(self.context.conversation_history)
            return history[-n:] if len(history) > n else history
    
    def get_context_snapshot(self) -> GameContext:
        """
        Get a snapshot of current context.
        
        Returns:
            Copy of current GameContext
        """
        with self._lock:
            # Create a shallow copy with deep copy of mutable fields
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
        Get a human-readable summary of current context.
        
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
        """
        Apply detected context changes.
        
        Args:
            changes: Dict of changes from detect_context_changes()
        """
        if "location" in changes:
            self.update_location(changes["location"])
        
        if "time" in changes:
            self.update_time(changes["time"])
        
        if "mood" in changes:
            self.update_mood(changes["mood"])

