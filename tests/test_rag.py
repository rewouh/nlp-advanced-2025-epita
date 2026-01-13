#!/usr/bin/env python3

import os
import sys
import tempfile
import pytest
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.models import (
    WorldLore,
    SessionState,
    NPCArchetype,
    NPCDefinition,
    Location,
    HistoricalEvent,
    Quest,
    Item,
    Faction,
    WorldMetadata,
)
from src.rag.lore_manager import (
    LoreManager,
    generate_quick_npc,
)


@pytest.fixture
def sample_world_yaml():
    return {
        "metadata": {
            "campaign_name": "Test Campaign",
            "setting": "fantasy",
        },
        "archetypes": [
            {
                "id": "tavern_keeper",
                "traits": ["friendly", "gossipy"],
                "knowledge": ["local_rumors", "geography"],
                "motives": "make money, keep peace",
            },
            {
                "id": "scholar",
                "traits": ["curious", "bookish"],
                "knowledge": ["ancient_history", "magic"],
                "motives": "learn secrets",
            },
        ],
        "npcs": [
            {
                "id": "joe",
                "name": "Joe",
                "archetype": "tavern_keeper",
                "traits": ["grumpy"],
                "location": "tavern",
                "backstory": "A former sailor.",
                "secrets": "Knows about the treasure.",
            }
        ],
        "locations": [
            {
                "id": "tavern",
                "name": "The Rusty Anchor",
                "description": "A dimly lit tavern with a fireplace and creaky floor.",
                "secrets": "Hidden tunnel in the cellar.",
            }
        ],
        "history": [
            {
                "event": "The Great War",
                "content": "A war that shaped the world. Many died and new kingdoms formed.",
            }
        ],
        "items": [
            {
                "id": "magic_sword",
                "name": "Excalibur",
                "description": "A legendary blade that glows and is unbreakable.",
            }
        ],
        "quests": [
            {
                "id": "find_treasure",
                "hook": "Joe knows about hidden treasure and will tell trusted patrons to find the map and dig at X.",
            }
        ],
        "factions": [
            {
                "id": "guild",
                "name": "Merchants Guild",
                "description": "Controls trade. Seeks profit above all else.",
            }
        ],
    }


@pytest.fixture
def sample_session_yaml():
    return {
        "current_time": "Evening",
        "current_location": "The Rusty Anchor",
        "current_mood": "Tense",
        "active_quests": ["find_treasure"],
        "world_events": ["Strange lights in the sky"],
    }


@pytest.fixture
def temp_world_file(sample_world_yaml):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_world_yaml, f)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_session_file(sample_session_yaml):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_session_yaml, f)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def lore_manager():
    return LoreManager(persist_directory=":memory:")


class TestModels:
    def test_npc_archetype_creation(self):
        archetype = NPCArchetype(
            id="test",
            traits=["brave", "kind"],
            knowledge=["history"],
            motives="help others",
        )
        assert archetype.id == "test"
        assert len(archetype.traits) == 2

    def test_npc_definition_creation(self):
        npc = NPCDefinition(
            id="hero",
            name="The Hero",
            archetype="warrior",
            traits=["brave"],
            location="castle",
        )
        assert npc.id == "hero"
        assert npc.archetype == "warrior"

    def test_location_creation(self):
        loc = Location(
            id="castle",
            name="The Grand Castle",
            description="A majestic fortress with towers and a moat.",
            secrets="Hidden passage behind the throne.",
        )
        assert loc.id == "castle"
        assert "fortress" in loc.description

    def test_historical_event_creation(self):
        event = HistoricalEvent(
            event="The Siege",
            content="A long siege of the castle. The castle fell after months.",
        )
        assert event.event == "The Siege"

    def test_session_state_defaults(self):
        state = SessionState()
        assert state.current_time == "day"
        assert state.current_mood == "neutral"
        assert state.active_quests == []

    def test_world_lore_from_dict(self, sample_world_yaml):
        world = WorldLore(**sample_world_yaml)
        assert world.metadata.campaign_name == "Test Campaign"
        assert len(world.archetypes) == 2
        assert len(world.npcs) == 1

    def test_world_lore_get_methods(self, sample_world_yaml):
        world = WorldLore(**sample_world_yaml)

        archetype = world.get_archetype("tavern_keeper")
        assert archetype is not None
        assert archetype.id == "tavern_keeper"

        npc = world.get_npc("joe")
        assert npc is not None
        assert npc.name == "Joe"

        loc = world.get_location("tavern")
        assert loc is not None
        assert loc.name == "The Rusty Anchor"

        assert world.get_archetype("nonexistent") is None


class TestYAMLLoading:
    def test_load_world_file(self, lore_manager, temp_world_file):
        world = lore_manager.load_world(temp_world_file)
        assert world.metadata.campaign_name == "Test Campaign"
        assert len(world.archetypes) == 2

    def test_load_session_file(self, lore_manager, temp_session_file):
        session = lore_manager.load_session(temp_session_file)
        assert session.current_time == "Evening"
        assert session.current_mood == "Tense"

    def test_load_actual_files(self, lore_manager):
        world_path = Path(__file__).parent.parent / "lore" / "world.yaml"
        session_path = Path(__file__).parent.parent / "lore" / "session.yaml"

        if world_path.exists():
            world = lore_manager.load_world(world_path)
            assert world.metadata.campaign_name == "The Frozen North"

        if session_path.exists():
            session = lore_manager.load_session(session_path)
            assert session.current_location == "The Rusty Anchor Tavern"


class TestChromaDBIntegration:
    def test_indexing_creates_documents(self, temp_world_file):
        manager = LoreManager(
            persist_directory=":memory:", collection_name="test_indexing"
        )

        assert manager.collection.count() == 0
        manager.load_world(temp_world_file)
        assert manager.collection.count() > 0

    def test_retrieve_returns_results(self, lore_manager, temp_world_file):
        lore_manager.load_world(temp_world_file)

        results = lore_manager.retrieve("war history", n_results=3)
        assert len(results) > 0
        assert any("war" in r.lower() for r in results)

    def test_metadata_filtering(self, lore_manager, temp_world_file):
        lore_manager.load_world(temp_world_file)

        results_joe = lore_manager.retrieve(
            "Tell me about ancient wars", npc_id="joe", n_results=5
        )

        results_scholar = lore_manager.retrieve(
            "Tell me about ancient wars", archetype_id="scholar", n_results=5
        )

        assert isinstance(results_joe, list)
        assert isinstance(results_scholar, list)

    def test_build_context_includes_session(
        self, lore_manager, temp_world_file, temp_session_file
    ):
        lore_manager.load_world(temp_world_file)
        lore_manager.load_session(temp_session_file)

        context = lore_manager.build_context_for_npc("Hello", npc_id="joe")

        assert "CURRENT SITUATION" in context
        assert "Evening" in context
        assert "Tense" in context


class TestQuickNPCGeneration:
    def test_generate_quick_npc(self, lore_manager, temp_world_file, temp_session_file):
        lore_manager.load_world(temp_world_file)
        lore_manager.load_session(temp_session_file)

        npc_input = generate_quick_npc(lore_manager, "tavern_keeper")

        assert npc_input.npc_id.endswith("_tavern_keeper")
        assert len(npc_input.persona.traits) >= 1
        assert npc_input.global_state.location == "The Rusty Anchor"

    def test_generate_quick_npc_with_name(
        self, lore_manager, temp_world_file, temp_session_file
    ):
        lore_manager.load_world(temp_world_file)
        lore_manager.load_session(temp_session_file)

        npc_input = generate_quick_npc(lore_manager, "tavern_keeper", name="Thorin")

        assert "Thorin" in npc_input.npc_id

    def test_generate_npc_without_world_data_raises(self, lore_manager):
        with pytest.raises(ValueError, match="World data not loaded"):
            generate_quick_npc(lore_manager, "tavern_keeper")


class TestE2EPipeline:
    def test_full_rag_to_orchestrator_flow(
        self, lore_manager, temp_world_file, temp_session_file
    ):
        lore_manager.load_world(temp_world_file)
        lore_manager.load_session(temp_session_file)

        npc_def = lore_manager.get_npc_definition("joe")
        assert npc_def is not None

        player_query = "Tell me about this place"
        context = lore_manager.build_context_for_npc(query=player_query, npc_id="joe")

        assert "CURRENT SITUATION" in context
        assert "The Rusty Anchor" in context
        assert isinstance(context, str)
        assert len(context) > 50

    def test_quick_npc_compatible_with_orchestrator(
        self, lore_manager, temp_world_file, temp_session_file
    ):
        lore_manager.load_world(temp_world_file)
        lore_manager.load_session(temp_session_file)

        npc_input = generate_quick_npc(lore_manager, "tavern_keeper")

        assert hasattr(npc_input, "npc_id")
        assert hasattr(npc_input, "persona")
        assert hasattr(npc_input, "global_state")
        assert hasattr(npc_input, "player_input")
        assert hasattr(npc_input, "retrieved_lore")

        assert len(npc_input.persona.traits) > 0


def run_all_tests():
    print("\n" + "=" * 60)
    print("RAG MODULE - TEST SUITE")
    print("=" * 60)
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_all_tests()
