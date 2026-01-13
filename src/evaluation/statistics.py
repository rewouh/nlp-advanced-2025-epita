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

def plot_response_times(tests_results):
    test_stats = []
    
    for tr in tests_results:
        test_name = tr['test_config'].path.name if tr['test_config'].path else "unknown"
        metrics = tr.get('metrics')
        if metrics and metrics.num_responses > 0:
            test_stats.append({
                'name': test_name,
                'avg_time': metrics.avg_response_time_ms,
                'total_time': metrics.total_response_time_ms,
                'num_responses': metrics.num_responses,
                'response_times': [r.response_time_ms for r in metrics.responses]
            })
    
    test_stats.sort(key=lambda x: x['avg_time'])
    
    test_names = [stat['name'] for stat in test_stats]
    avg_times = [stat['avg_time'] for stat in test_stats]
    
    colors = ['#388e3c' if t < 1000 else '#ffa726' if t < 2000 else '#d32f2f' for t in avg_times]
    
    plt.figure(figsize=(12, max(8, len(test_names) * 0.3)))
    bars = plt.barh(test_names, avg_times, color=colors)
    
    for i, (bar, time, stat) in enumerate(zip(bars, avg_times, test_stats)):
        plt.text(time + 50, i, f'{time:.0f}ms ({stat["num_responses"]} resp)', 
                va='center', fontsize=8)
    
    plt.xlabel('average response time (ms)', fontsize=12)
    plt.ylabel('test case', fontsize=12)
    plt.title('npc average response time per test', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FOLDER / "response_times.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_response_time_distribution(tests_results):
    all_response_times = []
    
    for tr in tests_results:
        metrics = tr.get('metrics')
        if metrics:
            all_response_times.extend([r.response_time_ms for r in metrics.responses])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.hist(all_response_times, bins=30, color='#66b3ff', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(all_response_times), color='red', linestyle='--', 
               label=f'Mean: {np.mean(all_response_times):.0f}ms')
    ax1.axvline(np.median(all_response_times), color='green', linestyle='--', 
               label=f'Median: {np.median(all_response_times):.0f}ms')
    ax1.set_xlabel('response time (ms)', fontsize=12)
    ax1.set_ylabel('frequency', fontsize=12)
    ax1.set_title('response time distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    ax2.boxplot(all_response_times, vert=True)
    ax2.set_ylabel('response time (ms)', fontsize=12)
    ax2.set_title('response time box plot', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    stats_text = f'count: {len(all_response_times)}\n'
    stats_text += f'mean: {np.mean(all_response_times):.1f}ms\n'
    stats_text += f'median: {np.median(all_response_times):.1f}ms\n'
    stats_text += f'std: {np.std(all_response_times):.1f}ms\n'
    stats_text += f'min: {np.min(all_response_times):.1f}ms\n'
    stats_text += f'max: {np.max(all_response_times):.1f}ms'
    ax2.text(1.15, 0.5, stats_text, transform=ax2.transAxes, 
            fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(FOLDER / "response_time_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_response_time_per_npc(tests_results):
    npc_stats = defaultdict(lambda: {'times': [], 'count': 0})
    
    for tr in tests_results:
        npc_id = tr['test_config'].npc_id
        metrics = tr.get('metrics')
        if metrics:
            for response in metrics.responses:
                npc_stats[npc_id]['times'].append(response.response_time_ms)
                npc_stats[npc_id]['count'] += 1
    
    npc_data = []
    for npc_id, stats in npc_stats.items():
        avg_time = np.mean(stats['times']) if stats['times'] else 0
        npc_data.append((npc_id, avg_time, stats['count']))
    
    npc_data.sort(key=lambda x: x[1])
    
    npc_ids = [data[0] for data in npc_data]
    avg_times = [data[1] for data in npc_data]
    counts = [data[2] for data in npc_data]
    
    colors = ['#388e3c' if t < 1000 else '#ffa726' if t < 2000 else '#d32f2f' for t in avg_times]
    
    plt.figure(figsize=(12, max(8, len(npc_ids) * 0.3)))
    bars = plt.barh(npc_ids, avg_times, color=colors)
    
    for i, (bar, time, count) in enumerate(zip(bars, avg_times, counts)):
        plt.text(time + 50, i, f'{time:.0f}ms ({count} resp)', va='center', fontsize=8)
    
    plt.xlabel('average response time (ms)', fontsize=12)
    plt.ylabel('NPC', fontsize=12)
    plt.title('npc average response time per npc', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FOLDER / "response_time_per_npc.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_token_usage(tests_results):
    test_stats = []
    
    for tr in tests_results:
        test_name = tr['test_config'].path.name if tr['test_config'].path else "unknown"
        metrics = tr.get('metrics')
        if metrics and metrics.num_responses > 0:
            test_stats.append({
                'name': test_name,
                'total_tokens': metrics.total_tokens,
                'prompt_tokens': metrics.total_prompt_tokens,
                'completion_tokens': metrics.total_completion_tokens,
                'avg_tokens': metrics.total_tokens / metrics.num_responses
            })
    
    test_stats.sort(key=lambda x: x['total_tokens'])
    
    test_names = [stat['name'] for stat in test_stats]
    prompt_tokens = [stat['prompt_tokens'] for stat in test_stats]
    completion_tokens = [stat['completion_tokens'] for stat in test_stats]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(8, len(test_names) * 0.3)))
    
    y_pos = np.arange(len(test_names))
    ax1.barh(y_pos, prompt_tokens, color='#66b3ff', label='Prompt tokens')
    ax1.barh(y_pos, completion_tokens, left=prompt_tokens, color='#99ff99', label='Completion tokens')
    
    for i, stat in enumerate(test_stats):
        ax1.text(stat['total_tokens'] + 50, i, f"{stat['total_tokens']}", va='center', fontsize=8)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(test_names)
    ax1.set_xlabel('total tokens', fontsize=12)
    ax1.set_ylabel('test case', fontsize=12)
    ax1.set_title('token usage per test', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    avg_tokens = [stat['avg_tokens'] for stat in test_stats]
    colors = ['#388e3c' if t < 500 else '#ffa726' if t < 1000 else '#d32f2f' for t in avg_tokens]
    
    ax2.barh(y_pos, avg_tokens, color=colors)
    
    for i, avg in enumerate(avg_tokens):
        ax2.text(avg + 10, i, f'{avg:.0f}', va='center', fontsize=8)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(test_names)
    ax2.set_xlabel('average tokens per response', fontsize=12)
    ax2.set_title('avg token usage per response', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FOLDER / "token_usage.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_token_usage_per_npc(tests_results):
    npc_stats = defaultdict(lambda: {'prompt': 0, 'completion': 0, 'count': 0})
    
    for tr in tests_results:
        npc_id = tr['test_config'].npc_id
        metrics = tr.get('metrics')
        if metrics:
            npc_stats[npc_id]['prompt'] += metrics.total_prompt_tokens
            npc_stats[npc_id]['completion'] += metrics.total_completion_tokens
            npc_stats[npc_id]['count'] += metrics.num_responses
    
    npc_data = []
    for npc_id, stats in npc_stats.items():
        total = stats['prompt'] + stats['completion']
        avg = total / stats['count'] if stats['count'] > 0 else 0
        npc_data.append({
            'id': npc_id,
            'total': total,
            'prompt': stats['prompt'],
            'completion': stats['completion'],
            'avg': avg,
            'count': stats['count']
        })
    
    npc_data.sort(key=lambda x: x['total'])
    
    npc_ids = [data['id'] for data in npc_data]
    prompt_tokens = [data['prompt'] for data in npc_data]
    completion_tokens = [data['completion'] for data in npc_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(8, len(npc_ids) * 0.3)))
    
    y_pos = np.arange(len(npc_ids))
    ax1.barh(y_pos, prompt_tokens, color='#66b3ff', label='Prompt tokens')
    ax1.barh(y_pos, completion_tokens, left=prompt_tokens, color='#99ff99', label='Completion tokens')
    
    for i, data in enumerate(npc_data):
        ax1.text(data['total'] + 50, i, f"{data['total']} ({data['count']} resp)", va='center', fontsize=8)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(npc_ids)
    ax1.set_xlabel('total tokens', fontsize=12)
    ax1.set_ylabel('npc', fontsize=12)
    ax1.set_title('token usage per npc', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    avg_tokens = [data['avg'] for data in npc_data]
    colors = ['#388e3c' if t < 500 else '#ffa726' if t < 1000 else '#d32f2f' for t in avg_tokens]
    
    ax2.barh(y_pos, avg_tokens, color=colors)
    
    for i, avg in enumerate(avg_tokens):
        ax2.text(avg + 10, i, f'{avg:.0f}', va='center', fontsize=8)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(npc_ids)
    ax2.set_xlabel('average tokens per response', fontsize=12)
    ax2.set_title('avg token usage per response', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FOLDER / "token_usage_per_npc.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_token_distribution(tests_results):
    all_prompt_tokens = []
    all_completion_tokens = []
    all_total_tokens = []
    
    for tr in tests_results:
        metrics = tr.get('metrics')
        if metrics:
            for response in metrics.responses:
                all_prompt_tokens.append(response.prompt_tokens)
                all_completion_tokens.append(response.completion_tokens)
                all_total_tokens.append(response.total_tokens)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    ax1.hist(all_total_tokens, bins=30, color='#66b3ff', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(all_total_tokens), color='red', linestyle='--', 
               label=f'mean: {np.mean(all_total_tokens):.0f}')
    ax1.axvline(np.median(all_total_tokens), color='green', linestyle='--', 
               label=f'median: {np.median(all_total_tokens):.0f}')
    ax1.set_xlabel('total tokens', fontsize=12)
    ax1.set_ylabel('frequency', fontsize=12)
    ax1.set_title('total token distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    ax2.scatter(all_prompt_tokens, all_completion_tokens, alpha=0.5, color='#66b3ff')
    ax2.set_xlabel('prompt tokens', fontsize=12)
    ax2.set_ylabel('completion tokens', fontsize=12)
    ax2.set_title('prompt vs completion tokens', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    box_data = [all_prompt_tokens, all_completion_tokens, all_total_tokens]
    ax3.boxplot(box_data, labels=['prompt', 'completion', 'total'])
    ax3.set_ylabel('token count', fontsize=12)
    ax3.set_title('token usage box plots', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    stats_text = 'token statistics:\n\n'
    stats_text += f'prompt tokens:\n'
    stats_text += f'  mean: {np.mean(all_prompt_tokens):.1f}\n'
    stats_text += f'  median: {np.median(all_prompt_tokens):.1f}\n'
    stats_text += f'  std: {np.std(all_prompt_tokens):.1f}\n\n'
    stats_text += f'completion tokens:\n'
    stats_text += f'  mean: {np.mean(all_completion_tokens):.1f}\n'
    stats_text += f'  median: {np.median(all_completion_tokens):.1f}\n'
    stats_text += f'  std: {np.std(all_completion_tokens):.1f}\n\n'
    stats_text += f'total tokens:\n'
    stats_text += f'  mean: {np.mean(all_total_tokens):.1f}\n'
    stats_text += f'  median: {np.median(all_total_tokens):.1f}\n'
    stats_text += f'  std: {np.std(all_total_tokens):.1f}\n'
    stats_text += f'  min: {np.min(all_total_tokens)}\n'
    stats_text += f'  max: {np.max(all_total_tokens)}\n'
    stats_text += f'\ntotal responses: {len(all_total_tokens)}'
    
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, 
            fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            family='monospace')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(FOLDER / "token_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_statistics(tests_results):
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
    
    plot_response_times(tests_results)
    print("generated: response_times.png")
    plot_response_time_distribution(tests_results)
    print("generated: response_time_distribution.png")
    plot_response_time_per_npc(tests_results)
    print("generated: response_time_per_npc.png")
    
    plot_token_usage(tests_results)
    print("generated: token_usage.png")
    plot_token_usage_per_npc(tests_results)
    print("generated: token_usage_per_npc.png")
    plot_token_distribution(tests_results)
    print("generated: token_distribution.png")
