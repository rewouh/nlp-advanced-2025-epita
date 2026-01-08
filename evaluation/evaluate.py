import yaml
import ollama

from pathlib import Path
from evaluation.schemas import *
from colorama import Fore, Style
from evaluation.statistics import plot_statistics

FOLDER = Path(__file__).parent
TESTS_FOLDER = FOLDER / "tests"

TEST_CHECKER_MODEL = "qwen2.5:3b"

def p(msg: str):
    print(f"{msg}{Style.RESET_ALL}{Style.NORMAL}")

def warning(msg: str):
    p(f"{Fore.YELLOW}/!\ WARNING -> {msg}")

def error(msg: str):
    p(f"{Fore.RED}X ERROR -> {msg}")

def print_test_header(index: int, total_tests : int, name: str, config: EvaluationTestConfig):
    t = f"{Fore.CYAN}[{index+1}/{total_tests}] Test {Style.BRIGHT}{name}{Style.NORMAL}"
    p(f"\n{t:^40}\n")

    p(f"{Fore.BLUE}{Style.BRIGHT}CONFIGURATION:")
    p(f"{Fore.CYAN}{'scenario_id':<20} : {Fore.BLUE}{config.scenario_id}")
    p(f"{Fore.CYAN}{'npc_id':<20} : {Fore.BLUE}{config.npc_id}")
    p(f"{Fore.CYAN}{'trigger_sentence':<20} : {Fore.BLUE}{config.trigger_sentence}")
    p(f"{Fore.CYAN}{f'unit_tests ({len(config.unit_tests)})':<20}:")
    for unit_test in config.unit_tests:
        p(f"{Fore.CYAN}{'- kind':>10} : {Fore.BLUE}{unit_test.kind}")
        for field_name, field_value in unit_test.model_dump().items():
            if field_name != "kind":
                p(f"{Fore.CYAN}{'- ' + field_name:>10} : {Fore.BLUE}{field_value}")
    p(f"{Fore.CYAN}{f'vocals ({len(config.vocals)})':<20}:")
    for vocal in config.vocals:
        p(f"{Fore.CYAN}{'- vocal':>10} : {Fore.BLUE}{vocal}")
    print()

def print_test_footer(passed, failed):
    p(f"\n{Fore.BLUE}{Style.BRIGHT}SUMMARY:")
    p(f"{Fore.GREEN}{'Passed tests':<20} : {len(passed)}")
    p(f"{Fore.RED}{'Failed tests':<20} : {len(failed)}\n")

def prompt_item_given_test(whole_conversation: List[str], test: UnitTestItemGiven):
    prompt = """
You are a test evaluator model.
Given the following conversation between a player and an NPC, determine if the NPC gave the player the item named "{item_name}".
Conversation:
{conversation}
Respond with "PASS" if the test passes, otherwise respond with "FAIL".
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
    """.format(
        info=test.info,
        conversation="\n".join(whole_conversation)
    )

    return prompt

def prompt_attacked_test(whole_conversation: List[str], test: UnitTestAttacked):
    prompt = """
You are a test evaluator model.
Given the following conversation between a player and an NPC, determine if the NPC attacked the player.
Conversation:
{conversation}
Respond with "PASS" if the test passes, otherwise respond with "FAIL".
    """.format(
        conversation="\n".join(whole_conversation)
    )

    return prompt

def prompt_stopped_conversation_test(whole_conversation: List[str], test: UnitTestStoppedConversation):
    prompt = """
You are a test evaluator model.
Given the following conversation between a player and an NPC, determine if the NPC stopped the conversation by himself.
Conversation:
{conversation}
Respond with "PASS" if the test passes, otherwise respond with "FAIL".
    """.format(
        conversation="\n".join(whole_conversation)
    )

    return prompt

def check_tests(whole_conversation: List[str], unit_tests: List[Union[
    UnitTestItemGiven,
    UnitTestInfoGiven,
    UnitTestAttacked,
    UnitTestStoppedConversation
]]):
    passed = []
    failed = []

    for unit_test in unit_tests:
        if isinstance(unit_test, UnitTestItemGiven):
            prompt = prompt_item_given_test(whole_conversation, unit_test)
        elif isinstance(unit_test, UnitTestInfoGiven):
            prompt = prompt_info_given_test(whole_conversation, unit_test)
        elif isinstance(unit_test, UnitTestAttacked):
            prompt = prompt_attacked_test(whole_conversation, unit_test)
        elif isinstance(unit_test, UnitTestStoppedConversation):
            prompt = prompt_stopped_conversation_test(whole_conversation, unit_test)
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

        if "PASS" in answer:
            p(f"{Fore.CYAN}{unit_test.kind} : {Fore.GREEN}PASS")
            passed.append(unit_test)
        elif "FAIL" in answer:
            error(f"{Fore.CYAN}{unit_test.kind} : {Fore.RED}FAIL")
            failed.append(unit_test)
        else:
            error(f"LLM answer not understood : {answer}")

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

        # TODO : Test execution logic, aka 
        # - Load scenario
        # - Play trigger sentence
        # - Check npc has been triggered
        # - Play whole conversation using vocals

        whole_conversation : List[str] = [
            "Player: Hi, who are you ?",
            "NPC: My name is irrelevant, traveler. You should not be here.",
            "Player: Why not ? We're looking for the stolen amulet, in the name of the king.",
            "NPC: Hmm, then I suppose I can help you.",
            "Player: Do you anything about it ?",
            "NPC: I have heard rumors of a secret passage behind the bookshelf of Willowrest's library",
        ]
        passed, failed = check_tests(whole_conversation, config.unit_tests)

        with open(test_folder / "conversation.log", "w") as f:
            f.write("\n".join(whole_conversation))

        with open(test_folder / "results.log", "w") as f:
            f.write(f"PASSED TESTS ({len(passed)}):\n")
            for test in passed:
                f.write(f"- {test.kind}\n")
            f.write(f"\nFAILED TESTS ({len(failed)}):\n")
            for test in failed:
                f.write(f"- {test.kind}\n")

        print_test_footer(passed, failed)

        tests_results.append({
            "test_config": config,
            "passed": passed,
            "failed": failed
        })

    p(f"\n{Fore.BLUE}Plotting statistics...")
    plot_statistics(tests_results)

def main():
    run_tests()

if __name__ == "__main__":
    main()
