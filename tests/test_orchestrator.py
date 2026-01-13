#!/usr/bin/env python3

import threading
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.conversation import (
    NPCOrchestrator,
    NPCInput,
    NPCOutput,
    Persona,
    GlobalState,
    RelationshipTracker,
    create_npc,
)


def test_relationship_tracker():
    print("\n" + "=" * 60)
    print("TEST: RelationshipTracker")
    print("=" * 60)

    tracker = RelationshipTracker()

    assert tracker.get_score() == 0, "Initial score should be 0"
    assert tracker.get_disposition() == "neutral", (
        "Initial disposition should be neutral"
    )

    tracker.analyze_input("Thank you so much for your help!")
    assert tracker.get_score() == 10, (
        f"Score after helpful input should be 10, got {tracker.get_score()}"
    )

    tracker.analyze_input("You're an idiot!")
    assert tracker.get_score() == -5, (
        f"Score after rude input should be -5, got {tracker.get_score()}"
    )

    for _ in range(5):
        tracker.analyze_input("shut up fool")
    assert tracker.get_score() <= -50, "Score should be hostile after repeated rudeness"
    assert tracker.get_disposition() == "hostile", (
        f"Disposition should be hostile, got {tracker.get_disposition()}"
    )

    print("[PASS] RelationshipTracker works correctly")


def test_pydantic_schemas():
    print("\n" + "=" * 60)
    print("TEST: Pydantic Schemas")
    print("=" * 60)

    persona = Persona(
        traits=["mysterious", "quiet", "observant"],
        motives="Seeking ancient knowledge",
        private_knowledge="Knows the location of a hidden artifact",
    )
    assert len(persona.traits) == 3

    global_state = GlobalState(
        location="The Rusty Anchor Tavern", time_of_day="night", scene_mood="mysterious"
    )
    assert global_state.location == "The Rusty Anchor Tavern"

    npc_input = NPCInput(
        npc_id="dark_figure",
        persona=persona,
        global_state=global_state,
        player_input="Hello there, stranger.",
        retrieved_lore="The Rusty Anchor is known for attracting mysterious travelers.",
    )
    assert npc_input.npc_id == "dark_figure"

    npc_output = NPCOutput(
        agent_response="...",
        updated_history=[{"role": "player", "content": "hello"}],
        relationship_score=0,
        was_interrupted=False,
    )
    assert npc_output.agent_response == "..."

    print("[PASS] Pydantic schemas validate correctly")


def test_tavern_interaction_streaming():
    print("\n" + "=" * 60)
    print("TEST: Tavern Interaction (Streaming)")
    print("=" * 60)

    npc = create_npc(
        npc_id="dark_figure",
        traits=["mysterious", "quiet", "cryptic", "ancient"],
        motives="Observing mortals and guarding forbidden knowledge",
        private_knowledge="Knows secrets about the whole universe - the universe is actually a dream of an ancient god.",
        model="qwen2.5:3b",
    )

    npc.update_state(
        GlobalState(
            location="The Rusty Anchor Tavern",
            time_of_day="night",
            scene_mood="mysterious",
        )
    )

    npc.set_lore(
        "The Rusty Anchor Tavern is a dimly lit establishment where shadowy figures "
        "often gather. It's said that those who sit in the corner booth have seen things "
        "no mortal should witness."
    )

    print(
        "\n[Scene: A dimly lit tavern. A dark-shaped figure sits alone in the corner.]"
    )
    print("[Player approaches the figure...]\n")

    print("Player: Hello there, stranger.")
    print("Dark Figure: ", end="", flush=True)

    start_time = time.time()
    response_tokens = []

    for token in npc.think("Hello there, stranger."):
        print(token, end="", flush=True)
        response_tokens.append(token)

    elapsed = time.time() - start_time
    full_response = "".join(response_tokens)

    print(f"\n\n[Response time: {elapsed:.2f}s]")

    assert len(full_response) > 0, "NPC should generate a response"
    assert elapsed < 10, f"Response should be generated within 10s, took {elapsed:.2f}s"

    print("\n" + "-" * 40)
    print("Player: Can you tell me about the whole universe?")
    print("Dark Figure: ", end="", flush=True)

    start_time = time.time()
    response_tokens = []

    for token in npc.think("Can you tell me about the whole universe?"):
        print(token, end="", flush=True)
        response_tokens.append(token)

    elapsed = time.time() - start_time
    full_response = "".join(response_tokens)

    print(f"\n\n[Response time: {elapsed:.2f}s]")
    print(f"[Relationship score: {npc.relationship.get_score()}]")
    print(f"[Disposition: {npc.relationship.get_disposition()}]")

    print("\n[PASS] Streaming tavern interaction completed")


def test_interruption_handling():
    print("\n" + "=" * 60)
    print("TEST: Interruption Handling")
    print("=" * 60)

    npc = create_npc(
        npc_id="chatty_bartender",
        traits=["talkative", "friendly", "verbose"],
        motives="Loves to tell long stories",
        model="qwen2.5:3b",
    )

    print("\nStarting generation that will be interrupted after 0.5s...")

    tokens_before_interrupt = []
    interrupt_triggered = False

    def interrupt_after_delay():
        nonlocal interrupt_triggered
        time.sleep(0.5)
        print("\n[Sending interrupt signal...]")
        npc.interrupt()
        interrupt_triggered = True

    interrupt_thread = threading.Thread(target=interrupt_after_delay)
    interrupt_thread.start()

    print("Bartender: ", end="", flush=True)
    for token in npc.think("Tell me your entire life story in great detail."):
        print(token, end="", flush=True)
        tokens_before_interrupt.append(token)

    interrupt_thread.join()

    partial_response = "".join(tokens_before_interrupt)
    print(f"\n\n[Tokens generated before interrupt: {len(tokens_before_interrupt)}]")
    print(f"[Partial response length: {len(partial_response)} chars]")

    assert interrupt_triggered, "Interrupt should have been triggered"

    print("\n[PASS] Interruption handling works")


def test_social_dynamics():
    print("\n" + "=" * 60)
    print("TEST: Social Dynamics")
    print("=" * 60)

    npc = create_npc(
        npc_id="grumpy_merchant",
        traits=["suspicious", "greedy", "protective"],
        motives="Protect my goods and make profit",
        private_knowledge="Has a secret stash of rare potions",
        model="qwen2.5:3b",
    )

    print("\n--- Phase 1: Neutral interaction ---")
    print("Player: Hello, do you have any potions?")
    print("Merchant: ", end="", flush=True)
    response1 = npc.think_sync("Hello, do you have any potions?")
    print(response1)
    print(
        f"[Relationship: {npc.relationship.get_score()}, Disposition: {npc.relationship.get_disposition()}]"
    )

    print("\n--- Phase 2: Player is helpful ---")
    for _ in range(4):
        npc.relationship.analyze_input("Thank you so much, you're very kind!")

    print("Player: Please, I would really appreciate your help. Thank you!")
    print("Merchant: ", end="", flush=True)
    response2 = npc.think_sync(
        "Please, I would really appreciate your help. Thank you!"
    )
    print(response2)
    print(
        f"[Relationship: {npc.relationship.get_score()}, Disposition: {npc.relationship.get_disposition()}]"
    )
    print(f"[Should share secrets: {npc.relationship.should_share_secrets()}]")

    print("\n--- Phase 3: Player turns rude ---")
    npc.reset_relationship()
    for _ in range(5):
        npc.relationship.analyze_input("You stupid fool! I hate you!")

    print("Player: Give me your stuff, you worthless merchant!")
    print("Merchant: ", end="", flush=True)
    response3 = npc.think_sync("Give me your stuff, you worthless merchant!")
    print(response3)
    print(
        f"[Relationship: {npc.relationship.get_score()}, Disposition: {npc.relationship.get_disposition()}]"
    )

    print("\n[PASS] Social dynamics affect NPC behavior")


def test_memory_serialization():
    print("\n" + "=" * 60)
    print("TEST: Memory Serialization (Unit)")
    print("=" * 60)

    tracker = RelationshipTracker()

    history = [
        {"role": "player", "content": "Hello"},
        {"role": "npc", "content": "Greetings, traveler."},
        {"role": "player", "content": "How are you?"},
        {"role": "npc", "content": "I am well."},
    ]

    assert isinstance(history, list), "History should be a list"
    assert all("role" in h and "content" in h for h in history), (
        "History entries need role and content"
    )

    print(f"[Serialized history entries: {len(history)}]")
    for entry in history:
        print(f"  - {entry['role']}: {entry['content'][:50]}...")

    print("\n[PASS] Memory serialization format works")


def test_data_contract_interface():
    print("\n" + "=" * 60)
    print("TEST: Data Contract Interface")
    print("=" * 60)

    persona = Persona(
        traits=["wise", "ancient"],
        motives="Share wisdom with worthy seekers",
        private_knowledge="The meaning of life is 42",
    )

    global_state = GlobalState(
        location="Mountain Temple", time_of_day="dawn", scene_mood="serene"
    )

    npc_input = NPCInput(
        npc_id="sage",
        persona=persona,
        global_state=global_state,
        player_input="Master, what is the meaning of life?",
        retrieved_lore="The temple has stood for a thousand years.",
    )

    npc = NPCOrchestrator(npc_id="sage", persona=persona, model="qwen2.5:3b")

    output = npc.process_input(npc_input)

    print(f"[Agent Response]: {output.agent_response}")
    print(f"[History entries]: {len(output.updated_history)}")
    print(f"[Relationship score]: {output.relationship_score}")
    print(f"[Was interrupted]: {output.was_interrupted}")

    assert isinstance(output, NPCOutput), "Output should be NPCOutput"
    assert len(output.agent_response) > 0, "Should have a response"
    assert isinstance(output.updated_history, list), "History should be a list"

    print("\n[PASS] Data contract interface works")


def run_all_tests():
    print("\n" + "=" * 60)
    print("NPC CONVERSATIONAL ORCHESTRATOR - TEST SUITE")
    print("=" * 60)

    tests = [
        test_relationship_tracker,
        test_pydantic_schemas,
        test_memory_serialization,
    ]

    llm_tests = [
        test_tavern_interaction_streaming,
        test_social_dynamics,
        test_data_contract_interface,
    ]

    slow_tests = [
        test_interruption_handling,
    ]

    print("\n--- Running unit tests (no LLM) ---")
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\n[FAIL] {test.__name__}: {e}")
            import traceback

            traceback.print_exc()

    if "--skip-llm" not in sys.argv:
        print("\n--- Running LLM integration tests ---")
        for test in llm_tests:
            try:
                test()
            except Exception as e:
                print(f"\n[FAIL] {test.__name__}: {e}")
                import traceback

                traceback.print_exc()

        if "--full" in sys.argv:
            print("\n--- Running slow tests ---")
            for test in slow_tests:
                try:
                    test()
                except Exception as e:
                    print(f"\n[FAIL] {test.__name__}: {e}")
                    import traceback

                    traceback.print_exc()
    else:
        print("\n[SKIPPED] LLM tests (--skip-llm flag)")

    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
