"""
LoreManager - RAG pipeline for world lore retrieval.

Handles YAML loading, document chunking, embedding, and retrieval
with metadata filtering based on NPC archetypes.
"""

import random
import yaml
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, cast

import chromadb

from src.rag.models import (
    WorldLore,
    SessionState,
    NPCArchetype,
    NPCDefinition,
)

from src.conversation import Persona, GlobalState, NPCInput


class LoreManager:
    """Manages world lore retrieval using ChromaDB."""

    CATEGORY_MAPPINGS = {
        "history": ["ancient_history", "local_history", "lore"],
        "geography": ["geography", "locations", "trade_routes"],
        "local_rumors": ["local_rumors", "gossip", "dock_gossip"],
        "local_laws": ["local_laws", "recent_crimes"],
        "magic": ["magic", "arcane", "spells"],
        "religion": ["religion", "deities", "rituals"],
        "secrets": ["secrets", "hidden_knowledge"],
        "items": ["weapons", "armor", "item_prices", "metallurgy"],
        "factions": ["factions", "organizations"],
    }

    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        collection_name: str = "world_lore",
        use_ollama_embeddings: bool = True,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.use_ollama_embeddings = use_ollama_embeddings

        if persist_directory == ":memory:":
            self.client = chromadb.Client()
        else:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_directory)

        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

        self.world_data: Optional[WorldLore] = None
        self.session_state: Optional[SessionState] = None
        self._indexed = False

    def load_world(self, world_path: Union[str, Path]) -> WorldLore:
        world_path = Path(world_path)

        with open(world_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        self.world_data = WorldLore(**raw_data)

        if self.collection.count() == 0:
            self._index_world_data()

        return self.world_data

    def load_session(self, session_path: Union[str, Path]) -> SessionState:
        session_path = Path(session_path)

        with open(session_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        self.session_state = SessionState(**raw_data)
        return self.session_state

    def update_session(self, session_state: SessionState) -> None:
        self.session_state = session_state

    def _index_world_data(self) -> None:
        if self.world_data is None:
            raise ValueError("World data not loaded. Call load_world() first.")

        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for i, event in enumerate(self.world_data.history):
            text = f"Historical Event: {event.event}. {event.content}"
            documents.append(text)
            metadatas.append({"category": "ancient_history", "type": "history"})
            ids.append(f"hist_{i}")

        for i, loc in enumerate(self.world_data.locations):
            text = f"Location: {loc.name}. {loc.description}"
            documents.append(text)
            metadatas.append(
                {
                    "category": "geography",
                    "type": "location",
                    "location_id": loc.id,
                }
            )
            ids.append(f"loc_{i}")

        for i, item in enumerate(self.world_data.items):
            text = f"Item: {item.name}. {item.description}"
            documents.append(text)
            metadatas.append(
                {
                    "category": "items",
                    "type": "item",
                    "item_id": item.id,
                }
            )
            ids.append(f"item_{i}")

        for i, quest in enumerate(self.world_data.quests):
            text = f"Quest: {quest.hook}"
            documents.append(text)
            metadatas.append(
                {
                    "category": "local_rumors",
                    "type": "quest",
                    "quest_id": quest.id,
                }
            )
            ids.append(f"quest_{i}")

        for i, faction in enumerate(self.world_data.factions):
            text = f"Faction: {faction.name}. {faction.description}"
            documents.append(text)
            metadatas.append(
                {
                    "category": "factions",
                    "type": "faction",
                    "faction_id": faction.id,
                }
            )
            ids.append(f"faction_{i}")

        for i, npc in enumerate(self.world_data.npcs):
            if npc.backstory:
                text = f"Person: {npc.name}. {npc.backstory}"
                documents.append(text)
                metadatas.append(
                    {
                        "category": "local_rumors",
                        "type": "npc_backstory",
                        "npc_id": npc.id,
                    }
                )
                ids.append(f"npc_{i}")

        if documents:
            self.collection.add(
                documents=documents, metadatas=cast(Any, metadatas), ids=ids
            )

        self._indexed = True

    def _get_knowledge_categories(
        self, npc_id: Optional[str] = None, archetype_id: Optional[str] = None
    ) -> List[str]:
        if self.world_data is None:
            return []

        categories: List[str] = []

        npc = None
        if npc_id:
            npc = self.world_data.get_npc(npc_id)

        archetype = None
        if npc and npc.archetype:
            archetype = self.world_data.get_archetype(npc.archetype)
        elif archetype_id:
            archetype = self.world_data.get_archetype(archetype_id)

        if archetype:
            categories.extend(archetype.knowledge)

        expanded = set()
        for cat in categories:
            expanded.add(cat)
            for key, values in self.CATEGORY_MAPPINGS.items():
                if cat in values or cat == key:
                    expanded.update(values)

        return list(expanded)

    def retrieve(
        self,
        query: str,
        npc_id: Optional[str] = None,
        archetype_id: Optional[str] = None,
        n_results: int = 3,
    ) -> List[str]:
        if self.collection.count() == 0:
            return []

        categories = self._get_knowledge_categories(npc_id, archetype_id)

        if categories:
            where_filter: Dict[str, Any] = {"category": {"$in": categories}}
            results = self.collection.query(
                query_texts=[query], n_results=n_results, where=where_filter
            )
        else:
            results = self.collection.query(query_texts=[query], n_results=n_results)

        if results and results["documents"] and results["documents"][0]:
            return results["documents"][0]

        return []

    def build_context_for_npc(
        self,
        query: str,
        npc_id: Optional[str] = None,
        archetype_id: Optional[str] = None,
        n_results: int = 3,
    ) -> str:
        parts = []

        if self.session_state:
            session_info = [
                f"Current Time: {self.session_state.current_time}",
                f"Location: {self.session_state.current_location}",
                f"Scene Mood: {self.session_state.current_mood}",
            ]
            if self.session_state.world_events:
                session_info.append(
                    f"Recent Events: {'; '.join(self.session_state.world_events)}"
                )

            parts.append("--- CURRENT SITUATION ---")
            parts.append("\n".join(session_info))

        lore_chunks = self.retrieve(query, npc_id, archetype_id, n_results)

        if lore_chunks:
            parts.append("\n--- RELEVANT KNOWLEDGE ---")
            for chunk in lore_chunks:
                parts.append(f"- {chunk}")

        return "\n".join(parts) if parts else "No specific context available."

    def get_npc_definition(self, npc_id: str) -> Optional[NPCDefinition]:
        if self.world_data:
            return self.world_data.get_npc(npc_id)
        return None

    def get_archetype(self, archetype_id: str) -> Optional[NPCArchetype]:
        if self.world_data:
            return self.world_data.get_archetype(archetype_id)
        return None

    def clear_index(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._indexed = False


def generate_quick_npc(
    manager: LoreManager,
    archetype_id: str,
    name: Optional[str] = None,
) -> NPCInput:
    """Generate a quick NPC from an archetype for improvised encounters."""
    if manager.world_data is None:
        raise ValueError("World data not loaded. Call manager.load_world() first.")

    archetype = manager.world_data.get_archetype(archetype_id)
    if archetype is None:
        raise ValueError(f"Archetype '{archetype_id}' not found.")

    npc_name = name or f"Unknown {archetype_id.replace('_', ' ').title()}"

    num_traits = min(len(archetype.traits), random.randint(2, 3))
    traits = random.sample(archetype.traits, num_traits) if archetype.traits else []

    global_state = GlobalState()
    if manager.session_state:
        global_state = GlobalState(
            location=manager.session_state.current_location,
            time_of_day=manager.session_state.current_time,
            scene_mood=manager.session_state.current_mood,
        )

    return NPCInput(
        npc_id=f"{npc_name}_{archetype_id}",
        persona=Persona(
            traits=traits,
            motives=archetype.motives,
            private_knowledge="",
        ),
        global_state=global_state,
        player_input="",
        retrieved_lore="",
    )
