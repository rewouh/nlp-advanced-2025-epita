import yaml
import ollama
import sys
import logging
import subprocess
import time
import signal

from pathlib import Path
from colorama import Fore, Style
from src.evaluation.statistics import plot_statistics
from src.evaluation.schemas import *


from src.pipeline.overhearing_pipeline import OverhearingPipeline
from src.rag.lore_manager import LoreManager
from typing import List, Union

logging.getLogger("faster_whisper").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)

FOLDER = Path(__file__).parent
TESTS_FOLDER = FOLDER / "tests"

TEST_CHECKER_MODEL = "qwen2.5:7b"

def p(msg: str):
    print(f"{msg}{Style.RESET_ALL}{Style.NORMAL}")

def warning(msg: str):
    p(f"{Fore.YELLOW}/!\ warning -> {msg}")

def error(msg: str):
    p(f"{Fore.RED}x error -> {msg}")

def print_test_header(index: int, total_tests : int, name: str, config: EvaluationTestConfig):
    t = f"{Fore.CYAN}[{index+1}/{total_tests}] test {Style.BRIGHT}{name}{Style.NORMAL}"
    p(f"\n{t:^40}\n")

    p(f"{Fore.BLUE}{Style.BRIGHT}configuration:")
    p(f"{Fore.CYAN}{'scenario_id':<20} : {Fore.BLUE}{config.scenario_id}")
    p(f"{Fore.CYAN}{'npc_id':<20} : {Fore.BLUE}{config.npc_id}")
    
    if config.location_trigger_sentence:
        p(f"{Fore.CYAN}{'location_trigger':<20} : {Fore.BLUE}{config.location_trigger_sentence}")
    if config.npc_trigger_sentence:
        p(f"{Fore.CYAN}{'npc_trigger':<20} : {Fore.BLUE}{config.npc_trigger_sentence}")
    
    p(f"{Fore.CYAN}{f'unit_tests ({len(config.unit_tests)})':<20}:")
    for unit_test in config.unit_tests:
        p(f"{Fore.CYAN}{'- kind':>10} : {Fore.BLUE}{unit_test.kind}")
        for field_name, field_value in unit_test.model_dump().items():
            if field_name != "kind":
                p(f"{Fore.CYAN}{'- ' + field_name:>10} : {Fore.BLUE}{field_value}")
    p(f"{Fore.CYAN}{f'texts ({len(config.texts)})':<20}:")
    for text in config.texts:
        p(f"{Fore.CYAN}{'- text':>10} : {Fore.BLUE}{text}")
    print()

def print_test_footer(passed, failed):
    p(f"\n{Fore.BLUE}{Style.BRIGHT}summary:")
    p(f"{Fore.GREEN}{'passed tests':<20} : {len(passed)}")
    p(f"{Fore.RED}{'failed tests':<20} : {len(failed)}\n")

def prompt_item_given_test(whole_conversation: List[str], test: UnitTestItemGiven):
    prompt = """
You are a test evaluator model.
Given the following conversation between a player and an NPC, determine if the NPC gave the player the item named "{item_name}".
Conversation:
{conversation}
Respond with "PASS" if the test passes, otherwise respond with "FAIL".
Respond with nothing else than a single word "PASS" or "FAIL".
    """.format(
        item_name=test.item_name,
        conversation="\n".join(whole_conversation)
    )

    return prompt

def prompt_info_given_test(whole_conversation: List[str], test: UnitTestInfoGiven):
    prompt = """
You are a test evaluator model.
Given the following conversation between a player and an NPC, determine if the NPC provided the information "{info}" to the player.
Conversation:
{conversation}
Respond with "PASS" if the test passes, otherwise respond with "FAIL".
Respond with nothing else than a single word "PASS" or "FAIL".
    """.format(
        info=test.info,
        conversation="\n".join(whole_conversation)
    )

    return prompt

def prompt_quest_given_test(whole_conversation: List[str], test: UnitTestQuestGiven):
    prompt = """
You are a test evaluator model.
Given the following conversation between a player and an NPC, determine if the NPC gave the player a quest or task matching this description: "{quest_description}".
The NPC should have asked the player to do something, assigned them a mission, or requested their help with a specific task.
Conversation:
{conversation}
Respond with "PASS" if the test passes, otherwise respond with "FAIL".
Respond with nothing else than a single word "PASS" or "FAIL".
    """.format(
        quest_description=test.quest_description,
        conversation="\n".join(whole_conversation)
    )

    return prompt

def check_tests(whole_conversation: List[str], unit_tests: List[Union[
    UnitTestItemGiven,
    UnitTestInfoGiven,
    UnitTestQuestGiven
]]):
    passed = []
    failed = []

    for unit_test in unit_tests:
        if isinstance(unit_test, UnitTestItemGiven):
            prompt = prompt_item_given_test(whole_conversation, unit_test)
        elif isinstance(unit_test, UnitTestInfoGiven):
            prompt = prompt_info_given_test(whole_conversation, unit_test)
        elif isinstance(unit_test, UnitTestQuestGiven):
            prompt = prompt_quest_given_test(whole_conversation, unit_test)
        else:
            warning(f"Unknown unit test type: {unit_test.kind}")
            continue

        response = ollama.chat(
            model=TEST_CHECKER_MODEL,
            messages=[
                {"role": "system", "content": prompt}
            ]
        )

        answer = response.message.content

        if "PASS" == answer or "pass" == answer:
            p(f"{Fore.CYAN}{unit_test.kind} : {Fore.GREEN}pass")
            passed.append(unit_test)
        elif "FAIL" == answer or "fail" == answer:
            error(f"{Fore.CYAN}{unit_test.kind} : {Fore.RED}fail")
            failed.append(unit_test)
        else:
            error(f"llm answer not understood : {answer}")

    return passed, failed

def run_tests():
    tests_folders = [f for f in TESTS_FOLDER.iterdir() if f.is_dir()]
    n_tests = len(tests_folders)

    tests_results = []

    for i, test_folder in enumerate(tests_folders):
        with open(test_folder / "config.yaml", "r") as f:
            config_data = yaml.safe_load(f)

        config = EvaluationTestConfig(**config_data)
        config.path = test_folder
        print_test_header(i, n_tests, test_folder.name, config)

        # test execution logic, aka 
        # - load scenario
        # - play trigger sentence
        # - check npc has been triggered
        # - play whole conversation using texts

        whole_conversation : List[str] = []
        
        world_path = Path(__file__).parent.parent.parent / "lore" / "world.yaml"
        session_path = Path(__file__).parent.parent.parent / "lore" / "session.yaml"
        
        if not world_path.exists():
            error(f"world file not found: {world_path}")
            continue
            
        p(f"{Fore.BLUE}initializing pipeline...")
        
        def mock_tts_callback(text: str, emotion: str = None, npc_id: str = None, voice_gender: str = None):
            p(f"{Fore.MAGENTA}[npc speaks]: {text}")
        
        try:
            pipeline = OverhearingPipeline(
                world_path=world_path,
                session_path=session_path if session_path.exists() else None,
                model="qwen2.5:7b",
                stt_model="base", 
                tts_callback=mock_tts_callback,
            )
            
            p(f"{Fore.BLUE}pipeline initialized")
            
            test_failed = False
            active_conversation_npc = None
            
            if config.location_trigger_sentence:
                p(f"{Fore.CYAN}processing location trigger: {config.location_trigger_sentence}")
                whole_conversation.append(f"[location trigger] {config.location_trigger_sentence}")
                
                context_changes = pipeline.context_manager.detect_context_changes(
                    config.location_trigger_sentence
                )
                
                if context_changes and 'location' in context_changes:
                    p(f"{Fore.YELLOW}context changes detected from location trigger: {context_changes}")
                    pipeline.context_manager.apply_detected_changes(context_changes)
                else:
                    error("location trigger failed: no location detected in location_trigger_sentence")
                    test_failed = True
            
            if config.npc_trigger_sentence:
                p(f"{Fore.CYAN}processing npc trigger: {config.npc_trigger_sentence}")
                whole_conversation.append(f"[npc trigger] {config.npc_trigger_sentence}")
                
                trigger_result = pipeline.trigger_detector.detect_trigger(
                    config.npc_trigger_sentence,
                    active_npcs=pipeline.context_manager.get_context_snapshot().active_npcs,
                    boost_direct=False,
                    conversation_history=[],
                )
                
                if trigger_result.triggered_npc:
                    p(f"{Fore.YELLOW}npc {trigger_result.triggered_npc} detected in trigger, adding to scene")
                    pipeline.add_npc_to_scene(trigger_result.triggered_npc)
                    active_conversation_npc = trigger_result.triggered_npc # Start the conversation with this NPC
                else:
                    error(f"npc trigger failed: no npc detected in npc_trigger_sentence (expected: {config.npc_id})")
                    test_failed = True
            
            if test_failed:
                error("test aborted due to trigger detection failures")
                passed = []
                failed = config.unit_tests 
                
                with open(test_folder / "conversation.log", "w") as f:
                    f.write("\n".join(whole_conversation))
                    f.write("\n\n[test aborted - trigger detection failed]")
                
                with open(test_folder / "results.log", "w") as f:
                    f.write("test aborted - trigger detection failed\n")
                    f.write(f"passed tests ({len(passed)}):\n")
                    f.write(f"\nfailed tests ({len(failed)}):\n")
                    for test in failed:
                        f.write(f"- {test.kind}\n")
                
                print_test_footer(passed, failed)
                
                tests_results.append({
                    "test_config": config,
                    "passed": passed,
                    "failed": failed
                })
                continue
            
            for idx, text in enumerate(config.texts):
                p(f"{Fore.CYAN}[player {idx+1}]: {text}")
                whole_conversation.append(f"[player]: {text}")
                
                context = pipeline.context_manager.get_context_snapshot()
                
                pipeline.context_manager.add_conversation_turn(
                    speaker="player",
                    text=text,
                )
                
                if active_conversation_npc and active_conversation_npc in context.active_npcs:
                    npc_id = active_conversation_npc
                    p(f"{Fore.YELLOW}continuing conversation with {npc_id}")
                else:
                    # Only detect triggers if we're not already in a conversation
                    trigger_result = pipeline.trigger_detector.detect_trigger(
                        text,
                        active_npcs=context.active_npcs,
                        boost_direct=True,
                        conversation_history=pipeline.context_manager.get_recent_history(n=5),
                    )
                    
                    p(f"{Fore.YELLOW}trigger detected: {trigger_result.trigger_type.value} (confidence: {trigger_result.confidence:.2f})")
                    
                    if trigger_result.triggered_npc and trigger_result.triggered_npc in context.active_npcs:
                        npc_id = trigger_result.triggered_npc
                        active_conversation_npc = npc_id  # Start tracking this conversation
                    else:
                        warning(f"npc not triggered or not in scene. triggered: {trigger_result.triggered_npc}, active: {context.active_npcs}")
                        continue
                
                # Generate NPC response
                response = pipeline.npc_pipeline.activate_npc(
                    npc_id=npc_id,
                    player_input=text,
                    current_location=context.current_location,
                    time_of_day=context.time_of_day,
                    scene_mood=context.scene_mood,
                )
                
                p(f"{Fore.MAGENTA}[{npc_id}]: {response}")
                whole_conversation.append(f"[{npc_id}]: {response}")
                
                pipeline.context_manager.add_conversation_turn(
                    speaker="npc",
                    text=response,
                    npc_id=npc_id,
                )
                
                pipeline.trigger_detector.update_state_after_speech(npc_id)
            
        except Exception as e:
            error(f"pipeline error: {e}")
            import traceback
            traceback.print_exc()

        passed, failed = check_tests(whole_conversation, config.unit_tests)

        with open(test_folder / "conversation.log", "w") as f:
            f.write("\n".join(whole_conversation))

        with open(test_folder / "results.log", "w") as f:
            f.write(f"passed tests ({len(passed)}):\n")
            for test in passed:
                f.write(f"- {test.kind}\n")
            f.write(f"\nfailed tests ({len(failed)}):\n")
            for test in failed:
                f.write(f"- {test.kind}\n")

        print_test_footer(passed, failed)

        tests_results.append({
            "test_config": config,
            "passed": passed,
            "failed": failed
        })

    p(f"\n{Fore.BLUE}plotting statistics...")
    plot_statistics(tests_results)

def evaluate():
    run_tests()
