#!/usr/bin/env python3
"""
Tests for STT Engine with hotwords support.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stt import STTEngine, HotwordExtractor, create_stt_engine
from src.rag.lore_manager import LoreManager


def test_hotword_extractor():
    """Test hotword extraction from world lore."""
    print("\n" + "=" * 60)
    print("TEST: Hotword Extraction")
    print("=" * 60)
    
    lore_manager = LoreManager(persist_directory=":memory:")
    world_path = Path(__file__).parent.parent / "lore" / "world.yaml"
    
    if not world_path.exists():
        print("[SKIP] world.yaml not found")
        return
    
    world = lore_manager.load_world(world_path)
    hotwords = HotwordExtractor.extract_from_world(world)
    
    print(f"\n[Extracted {len(hotwords)} hotwords]")
    print(f"Sample hotwords: {hotwords[:10]}")
    
    # Check for expected hotwords
    assert "Joe" in hotwords, "NPC name 'Joe' should be in hotwords"
    assert any("tavern" in h.lower() for h in hotwords), "Tavern location should be in hotwords"
    
    # Format for Whisper
    formatted = HotwordExtractor.format_for_whisper(hotwords)
    print(f"\n[Formatted for Whisper]: {formatted[:100]}...")
    
    assert len(formatted) > 0, "Formatted hotwords should not be empty"
    
    print("\n[PASS] Hotword extraction works")


def test_stt_engine_initialization():
    """Test STT engine initialization."""
    print("\n" + "=" * 60)
    print("TEST: STT Engine Initialization")
    print("=" * 60)
    
    # Test with custom hotwords
    custom_hotwords = ["Phandalin", "Tiamat", "Cryovain", "Joe"]
    engine = STTEngine.get_instance(
        model_size="tiny",  # Use tiny for faster testing
        device="auto",
        compute_type="auto",
        hotwords=custom_hotwords,
    )
    
    print(f"\n[Model]: {engine.model_size}")
    print(f"[Device]: {engine.device}")
    print(f"[Hotwords]: {len(engine.hotwords)}")
    print(f"[Using faster-whisper]: {engine.use_faster_whisper}")
    
    assert engine.model is not None, "Model should be loaded"
    assert len(engine.hotwords) == 4, "Should have 4 hotwords"
    
    # Test singleton pattern
    engine2 = STTEngine.get_instance()
    assert engine2 is engine, "Should return same instance (singleton)"
    
    print("\n[PASS] STT Engine initialized correctly")


def test_stt_with_world_lore():
    """Test creating STT engine from world lore."""
    print("\n" + "=" * 60)
    print("TEST: STT Engine with World Lore")
    print("=" * 60)
    
    lore_manager = LoreManager(persist_directory=":memory:")
    world_path = Path(__file__).parent.parent / "lore" / "world.yaml"
    
    if not world_path.exists():
        print("[SKIP] world.yaml not found")
        return
    
    world = lore_manager.load_world(world_path)
    
    # Reset singleton for this test
    STTEngine._instance = None
    
    engine = create_stt_engine(
        world_lore=world,
        model_size="tiny",
        device="auto",
    )
    
    print(f"\n[Hotwords from lore]: {len(engine.hotwords)}")
    print(f"Sample: {engine.hotwords[:5]}")
    
    assert len(engine.hotwords) > 0, "Should extract hotwords from lore"
    assert "Joe" in engine.hotwords or any("joe" in h.lower() for h in engine.hotwords)
    
    print("\n[PASS] STT Engine created from world lore")


def test_add_hotwords():
    """Test adding hotwords dynamically."""
    print("\n" + "=" * 60)
    print("TEST: Dynamic Hotword Management")
    print("=" * 60)
    
    # Reset singleton
    STTEngine._instance = None
    
    engine = STTEngine.get_instance(
        model_size="tiny",
        hotwords=["Dragon", "Sword"],
    )
    
    initial_count = len(engine.hotwords)
    print(f"\n[Initial hotwords]: {initial_count}")
    
    # Add new hotwords
    engine.add_hotwords(["Goblin", "Treasure", "Castle"])
    
    print(f"[After adding]: {len(engine.hotwords)}")
    
    assert len(engine.hotwords) == initial_count + 3
    assert "Goblin" in engine.hotwords
    
    # Set new hotwords (replace)
    engine.set_hotwords(["NewWord1", "NewWord2"])
    print(f"[After setting]: {len(engine.hotwords)}")
    
    assert len(engine.hotwords) == 2
    assert "NewWord1" in engine.hotwords
    
    print("\n[PASS] Dynamic hotword management works")


def run_all_tests():
    """Run all STT tests."""
    print("\n" + "=" * 60)
    print("STT ENGINE - TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_hotword_extractor,
        test_stt_engine_initialization,
        test_stt_with_world_lore,
        test_add_hotwords,
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\n[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

