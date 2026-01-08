from evaluation.schemas import *
import matplotlib.pyplot as plt

FOLDER = Path(__file__).parent

def plot_results_by_type(passed, failed):
    passed_kinds = [unit_test.kind for unit_test in passed]
    failed_kinds = [unit_test.kind for unit_test in failed]

    kind_labels = list(UnitTestType)
    passed_counts = [passed_kinds.count(kind) for kind in kind_labels]
    failed_counts = [failed_kinds.count(kind) for kind in kind_labels]
    x = range(len(kind_labels))
    plt.figure(figsize=(10, 6))
    plt.bar(x, passed_counts, width=0.4, label='Passed', color='green', align='center')
    plt.bar(x, failed_counts, width=0.4, label='Failed', color='red', align='edge')
    plt.xlabel('Test type')
    plt.ylabel('Number of tests')
    plt.title('Test results by type')
    plt.xticks(x, [kind.value for kind in kind_labels])
    plt.legend()
    plt.tight_layout()
    plt.savefig(FOLDER / "results_by_type.png")

def plot_statistics(tests_results):
    passed, failed = sum([tr['passed'] for tr in tests_results], []), sum([tr['failed'] for tr in tests_results], [])
    plot_results_by_type(passed, failed)
