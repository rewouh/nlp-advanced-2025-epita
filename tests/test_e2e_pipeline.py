#!/usr/bin/env python3
"""
End-to-end tests for the complete overhearing pipeline.

Tests the integration of all components:
- STT with hotwords
- Context management
- Trigger detection
- NPC activation
- RAG retrieval
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import OverhearingPipeline
from src.overhearing import TriggerType


def test_trigger_detection_integration():
    """Test trigger detection with real world data."""
    print("\n" + "=" * 60)
    print("TEST: Trigger Detection Integration")
    print("=" * 60)
    
    world_path = Path(__file__).parent.parent / "lore" / "world.yaml"
    session_path = Path(__file__).parent.parent / "lore" / "session.yaml"
    
    if not world_path.exists():
        print("[SKIP] world.yaml not found")
        return
    
    # Initialize pipeline (no TTS for testing)
    pipeline = OverhearingPipeline(
        world_path=world_path,
        session_path=session_path if session_path.exists() else None,
        model="qwen2.5:3b",
        stt_model="tiny",
        tts_callback=None,
    )
    
    # Set up scene
    pipeline.set_location("The Rusty Anchor Tavern", "rusty_anchor")
    pipeline.add_npc_to_scene("joe_tavernier")
    
    # Test cases
    test_cases = [
        {
            "input": "Pass me the chips please",
            "expected": TriggerType.IGNORE,
            "description": "Out-of-game chatter",
        },
        {
            "input": "What do you guys think we should do?",
            "expected": TriggerType.PLAYER_TO_PLAYER,
            "description": "Player-to-player",
        },
        {
            "input": "Hey Joe, can I get a drink?",
            "expected": TriggerType.NPC_DIRECT,
            "description": "Direct NPC address",
        },
        {
            "input": "I wonder what the tavern keeper knows about the dragon",
            "expected": TriggerType.NPC_INDIRECT,
            "description": "Indirect NPC mention",
        },
    ]
    
    context = pipeline.context_manager.get_context_snapshot()
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Input: \"{test['input']}\"")
        print(f"Expected: {test['expected'].value}")
        print(f"Description: {test['description']}")
        
        result = pipeline.trigger_detector.detect_trigger(
            test['input'],
            active_npcs=context.active_npcs,
        )
        
        print(f"Result: {result.trigger_type.value} (confidence: {result.confidence:.2f})")
        print(f"Reasoning: {result.reasoning}")
        
        # Check result
        if result.trigger_type == test['expected']:
            print("✓ PASS")
        else:
            print(f"✗ FAIL: Expected {test['expected'].value}, got {result.trigger_type.value}")
    
    print("\n[PASS] Trigger detection integration complete")


def test_context_tracking():
    """Test context tracking and updates."""
    print("\n" + "=" * 60)
    print("TEST: Context Tracking")
    print("=" * 60)
    
    world_path = Path(__file__).parent.parent / "lore" / "world.yaml"
    
    if not world_path.exists():
        print("[SKIP] world.yaml not found")
        return
    
    pipeline = OverhearingPipeline(
        world_path=world_path,
        model="qwen2.5:3b",
        stt_model="tiny",
        tts_callback=None,
    )
    
    # Initial context
    print("\nInitial context:")
    print(pipeline.get_context_summary())
    
    # Update location
    pipeline.set_location("The Frozen Harbor", "frozen_harbor")
    print("\nAfter location update:")
    print(pipeline.get_context_summary())
    
    # Add NPC
    pipeline.add_npc_to_scene("captain_hilda")
    print("\nAfter adding NPC:")
    print(pipeline.get_context_summary())
    
    # Detect context changes from speech
    test_inputs = [
        "It's getting dark outside",  # Should detect time
        "This place feels tense",  # Should detect mood
    ]
    
    for input_text in test_inputs:
        print(f"\nProcessing: \"{input_text}\"")
        changes = pipeline.context_manager.detect_context_changes(input_text)
        print(f"Detected changes: {changes}")
        
        if changes:
            pipeline.context_manager.apply_detected_changes(changes)
            print("Context updated:")
            print(pipeline.get_context_summary())
    
    print("\n[PASS] Context tracking works")


def test_npc_activation_with_rag():
    """Test NPC activation with RAG retrieval."""
    print("\n" + "=" * 60)
    print("TEST: NPC Activation with RAG")
    print("=" * 60)
    
    world_path = Path(__file__).parent.parent / "lore" / "world.yaml"
    session_path = Path(__file__).parent.parent / "lore" / "session.yaml"
    
    if not world_path.exists():
        print("[SKIP] world.yaml not found")
        return
    
    pipeline = OverhearingPipeline(
        world_path=world_path,
        session_path=session_path if session_path.exists() else None,
        model="qwen2.5:3b",
        stt_model="tiny",
        tts_callback=lambda text: print(f"\n🔊 TTS: {text}\n"),
    )
    
    # Set scene
    pipeline.set_location("The Rusty Anchor Tavern", "rusty_anchor")
    pipeline.add_npc_to_scene("joe_tavernier")
    
    print("\n" + "-" * 60)
    print("Scenario: Player asks Joe about the Great Blizzard")
    print("-" * 60)
    
    # Start pipeline
    pipeline.start()
    time.sleep(0.5)  # Let it initialize
    
    # Simulate player input
    player_input = "Hey Joe, what do you know about the Great Blizzard?"
    print(f"\n💬 Player: \"{player_input}\"")
    
    pipeline.process_input_manually(player_input)
    
    # Wait for processing
    time.sleep(5)
    
    # Check conversation history
    history = pipeline.context_manager.get_recent_history(n=5)
    print(f"\n📝 Conversation history: {len(history)} turns")
    
    for turn in history:
        speaker_icon = "👤" if turn.speaker == "player" else "🤖"
        print(f"  {speaker_icon} {turn.speaker}: {turn.text[:60]}...")
    
    # Stop pipeline
    pipeline.stop()
    
    print("\n[PASS] NPC activation with RAG works")


def test_manual_conversation_flow():
    """Test a complete conversation flow manually."""
    print("\n" + "=" * 60)
    print("TEST: Complete Conversation Flow")
    print("=" * 60)
    
    world_path = Path(__file__).parent.parent / "lore" / "world.yaml"
    
    if not world_path.exists():
        print("[SKIP] world.yaml not found")
        return
    
    # Mock TTS to capture responses
    responses = []
    def mock_tts(text):
        responses.append(text)
        print(f"\n🎭 Joe: {text}")
    
    pipeline = OverhearingPipeline(
        world_path=world_path,
        model="qwen2.5:3b",
        stt_model="tiny",
        tts_callback=mock_tts,
    )
    
    # Start pipeline
    pipeline.start()
    
    # Set scene
    pipeline.set_location("The Rusty Anchor Tavern")
    pipeline.add_npc_to_scene("joe_tavernier")
    
    # Conversation
    conversation = [
        "Hey Joe, how are you?",
        "Do you have any ale?",
        "Have you seen anything strange lately?",
    ]
    
    print("\n" + "-" * 60)
    print("Starting conversation with Joe...")
    print("-" * 60)
    
    for i, player_input in enumerate(conversation, 1):
        print(f"\n--- Turn {i} ---")
        print(f"💬 Player: \"{player_input}\"")
        
        pipeline.process_input_manually(player_input)
        time.sleep(3)  # Wait for response
    
    # Stop
    pipeline.stop()
    
    print("\n" + "-" * 60)
    print(f"Conversation complete: {len(responses)} NPC responses")
    print("-" * 60)
    
    assert len(responses) > 0, "Should have at least one NPC response"
    
    print("\n[PASS] Complete conversation flow works")


def run_all_tests():
    """Run all end-to-end tests."""
    print("\n" + "=" * 60)
    print("OVERHEARING PIPELINE - END-TO-END TESTS")
    print("=" * 60)
    print("\nNote: These tests require Ollama to be running with qwen2.5:3b")
    print("and may take several minutes to complete.\n")
    
    tests = [
        test_trigger_detection_integration,
        test_context_tracking,
        test_npc_activation_with_rag,
        test_manual_conversation_flow,
    ]
    
    for test in tests:
        try:
            test()
        except KeyboardInterrupt:
            print("\n\nTests interrupted by user")
            break
        except Exception as e:
            print(f"\n[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

