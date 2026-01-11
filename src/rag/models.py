"""
Pydantic models for RAG module data structures.

Defines the schema for world lore, NPC archetypes, session state,
and all related game entities.

Design principle: Only store what a DM would write in their notes.
Let the LLM improvise everything else (greetings, dialogue style, etc.)
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class NPCArchetype(BaseModel):
    """Template for quick NPC generation."""

    id: str = Field(..., description="Unique identifier for the archetype")
    traits: List[str] = Field(
        default_factory=list, description="Common personality traits for this archetype"
    )
    knowledge: List[str] = Field(
        default_factory=list,
        description="Categories of lore this archetype typically knows",
    )
    motives: str = Field(default="", description="What typically drives this archetype")


class NPCDefinition(BaseModel):
    """Fully defined NPC with custom attributes."""

    id: str = Field(..., description="Unique identifier for the NPC")
    name: str = Field(..., description="Display name of the NPC")
    archetype: Optional[str] = Field(
        default=None,
        description="Reference to base archetype (can be None for fully custom NPCs)",
    )
    traits: List[str] = Field(
        default_factory=list,
        description="Custom traits that override/extend archetype traits",
    )
    location: Optional[str] = Field(
        default=None, description="Default location where this NPC can be found"
    )
    backstory: str = Field(
        default="", description="NPC's personal history and background"
    )
    secrets: str = Field(
        default="", description="Secret information the NPC knows but may not share"
    )


class Location(BaseModel):
    """A place in the game world."""

    id: str = Field(..., description="Unique identifier for the location")
    name: str = Field(..., description="Display name of the location")
    description: str = Field(
        default="", description="General description of the location"
    )
    secrets: str = Field(
        default="", description="Hidden information about this location"
    )


class HistoricalEvent(BaseModel):
    """A historical event in the game world."""

    event: str = Field(..., description="Name of the event")
    content: str = Field(
        ..., description="Description of what happened and its effects"
    )


class Quest(BaseModel):
    """A quest or mission in the game - simplified to just the hook."""

    id: str = Field(..., description="Unique identifier for the quest")
    hook: str = Field(
        ..., description="How the quest is introduced (what NPCs might say)"
    )


class Item(BaseModel):
    """An item in the game world."""

    id: str = Field(..., description="Unique identifier for the item")
    name: str = Field(..., description="Display name of the item")
    description: str = Field(
        ..., description="Description of the item and its properties"
    )


class Faction(BaseModel):
    """An organization or group in the game world."""

    id: str = Field(..., description="Unique identifier for the faction")
    name: str = Field(..., description="Display name of the faction")
    description: str = Field(
        ..., description="Overview of the faction, goals, and relationships"
    )


class WorldMetadata(BaseModel):
    """Metadata about the campaign/world."""

    campaign_name: str = Field(..., description="Name of the campaign")
    setting: str = Field(default="fantasy", description="Genre/setting type")


class WorldLore(BaseModel):
    """Container for all static world data."""

    metadata: WorldMetadata
    archetypes: List[NPCArchetype] = Field(default_factory=list)
    npcs: List[NPCDefinition] = Field(default_factory=list)
    locations: List[Location] = Field(default_factory=list)
    history: List[HistoricalEvent] = Field(default_factory=list)
    items: List[Item] = Field(default_factory=list)
    quests: List[Quest] = Field(default_factory=list)
    factions: List[Faction] = Field(default_factory=list)

    def get_archetype(self, archetype_id: str) -> Optional[NPCArchetype]:
        """Get an archetype by ID."""
        return next((a for a in self.archetypes if a.id == archetype_id), None)

    def get_npc(self, npc_id: str) -> Optional[NPCDefinition]:
        """Get an NPC definition by ID."""
        return next((n for n in self.npcs if n.id == npc_id), None)

    def get_location(self, location_id: str) -> Optional[Location]:
        """Get a location by ID."""
        return next((loc for loc in self.locations if loc.id == location_id), None)


class SessionState(BaseModel):
    """Current game session state (dynamic, not embedded)."""

    current_time: str = Field(
        default="day", description="Current time of day (morning, day, evening, night)"
    )
    current_location: str = Field(
        default="unknown", description="Current location ID or name"
    )
    current_mood: str = Field(
        default="neutral",
        description="Current scene mood (tense, relaxed, mysterious, etc.)",
    )
    active_quests: List[str] = Field(
        default_factory=list, description="IDs of currently active quests"
    )
    world_events: List[str] = Field(
        default_factory=list, description="Current events happening in the world"
    )
