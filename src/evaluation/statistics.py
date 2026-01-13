from src.evaluation.schemas import *
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

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
    plt.xlabel('test type')
    plt.ylabel('number of tests')
    plt.title('test results by type')
    plt.xticks(x, [kind.value for kind in kind_labels])
    plt.legend()
    plt.tight_layout()
    plt.savefig(FOLDER / "results_by_type.png")
    plt.close()

def plot_success_rate_per_test(tests_results):
    test_stats = []
    
    for tr in tests_results:
        test_name = tr['test_config'].path.name if tr['test_config'].path else "unknown"
        total_tests = len(tr['passed']) + len(tr['failed'])
        success_rate = len(tr['passed']) / total_tests * 100 if total_tests > 0 else 0
        test_stats.append((test_name, success_rate, total_tests))
    
    test_stats.sort(key=lambda x: x[1])
    
    test_names = [stat[0] for stat in test_stats]
    success_rates = [stat[1] for stat in test_stats]
    
    colors = ['#d32f2f' if sr < 50 else '#ffa726' if sr < 80 else '#388e3c' for sr in success_rates]
    
    plt.figure(figsize=(12, max(8, len(test_names) * 0.3)))
    bars = plt.barh(test_names, success_rates, color=colors)
    
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        plt.text(rate + 1, i, f'{rate:.1f}%', va='center', fontsize=8)
    
    plt.xlabel('success rate (%)', fontsize=12)
    plt.ylabel('test case', fontsize=12)
    plt.title('success rate per test case', fontsize=14, fontweight='bold')
    plt.xlim(0, 110)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FOLDER / "success_rate_per_test.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_success_rate_per_npc(tests_results):
    npc_stats = defaultdict(lambda: {'passed': 0, 'failed': 0})
    
    for tr in tests_results:
        npc_id = tr['test_config'].npc_id
        npc_stats[npc_id]['passed'] += len(tr['passed'])
        npc_stats[npc_id]['failed'] += len(tr['failed'])
    
    npc_data = []
    for npc_id, stats in npc_stats.items():
        total = stats['passed'] + stats['failed']
        success_rate = stats['passed'] / total * 100 if total > 0 else 0
        npc_data.append((npc_id, success_rate, total))
    
    npc_data.sort(key=lambda x: x[1])
    
    npc_ids = [data[0] for data in npc_data]
    success_rates = [data[1] for data in npc_data]
    totals = [data[2] for data in npc_data]
    
    colors = ['#d32f2f' if sr < 50 else '#ffa726' if sr < 80 else '#388e3c' for sr in success_rates]
    
    plt.figure(figsize=(12, max(8, len(npc_ids) * 0.3)))
    bars = plt.barh(npc_ids, success_rates, color=colors)
    
    for i, (bar, rate, total) in enumerate(zip(bars, success_rates, totals)):
        plt.text(rate + 1, i, f'{rate:.1f}% ({total} tests)', va='center', fontsize=8)
    
    plt.xlabel('success rate (%)', fontsize=12)
    plt.ylabel('NPC', fontsize=12)
    plt.title('success rate per NPC', fontsize=14, fontweight='bold')
    plt.xlim(0, 110)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FOLDER / "success_rate_per_npc.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_test_type_distribution(passed, failed):
    all_tests = passed + failed
    kind_counts = defaultdict(int)
    
    for test in all_tests:
        kind_counts[test.kind.value] += 1
    
    labels = list(kind_counts.keys())
    sizes = list(kind_counts.values())
    colors = ['#66b3ff', '#99ff99', '#ffcc99']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('distribution of test types', fontsize=14, fontweight='bold')
    
    kind_labels = list(UnitTestType)
    passed_counts = [sum(1 for t in passed if t.kind == kind) for kind in kind_labels]
    failed_counts = [sum(1 for t in failed if t.kind == kind) for kind in kind_labels]
    
    success_rates = []
    for p, f in zip(passed_counts, failed_counts):
        total = p + f
        success_rates.append(p / total * 100 if total > 0 else 0)
    
    bars = ax2.bar([k.value for k in kind_labels], success_rates, 
                   color=['#388e3c' if sr >= 80 else '#ffa726' if sr >= 50 else '#d32f2f' 
                          for sr in success_rates])
    ax2.set_ylabel('success rate (%)', fontsize=12)
    ax2.set_xlabel('test type', fontsize=12)
    ax2.set_title('success rate by test type', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FOLDER / "test_type_distribution.png", dpi=300)
    plt.close()

def plot_statistics(tests_results):
    if not tests_results:
        print("No test results to plot")
        return
    
    passed = sum([tr['passed'] for tr in tests_results], [])
    failed = sum([tr['failed'] for tr in tests_results], [])
    
    plot_results_by_type(passed, failed)
    print("generated: results_by_type.png")
    plot_success_rate_per_test(tests_results)
    print("generated: success_rate_per_test.png")
    plot_success_rate_per_npc(tests_results)
    print("generated: success_rate_per_npc.png")
    plot_test_type_distribution(passed, failed)
    print("generated: test_type_distribution.png")
