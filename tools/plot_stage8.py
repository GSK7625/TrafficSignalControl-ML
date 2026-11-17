"""Tao cac bieu do tong hop Matplotlib tu cac file CSV dau ra cua Stage 8."""

import argparse
import csv
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


VARIANTS = ('QL-Original', 'QL-NewState', 'QL-NewState-Guard')
COLORS = {
    'QL-Original': '#4c78a8',
    'QL-NewState': '#e45756',
    'QL-NewState-Guard': '#54a24b',
}


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as file:
        return list(csv.DictReader(file))


def _grouped_bar(ax, runs, partition, metric, title, ylabel, scale=1.0, ylim=None):
    subset = [r for r in runs if r['partition'] == partition]
    for idx, label in enumerate(VARIANTS):
        vals = [float(r[metric]) * scale for r in subset if r['label'] == label]
        mean = statistics.mean(vals)
        low, high = mean - min(vals), max(vals) - mean
        ax.bar(idx, mean, yerr=[[low], [high]], width=0.58,
               color=COLORS[label], capsize=5, error_kw={'lw': 1.3})
        offsets = [0.0] if len(vals) == 1 else [0.24 * i / (len(vals) - 1) - 0.12 for i in range(len(vals))]
        ax.scatter([idx + off for off in offsets], vals, color='#1f1f1f', s=18, zorder=3)
        ax.text(idx, max(vals) + (ylim[1] * 0.025 if ylim else max(vals) * 0.04),
                f'{mean:.1f}' if scale == 1.0 else f'{mean:.2f}%',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.set_xticks(range(len(VARIANTS)), ['Original', 'NewState', 'NewState\n+ Guard'])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10.5)
    ax.set_ylim(ylim if ylim else (0, max(float(r[metric]) * scale for r in subset) * 1.18))
    ax.grid(axis='y', alpha=0.25)
    ax.set_axisbelow(True)


def plot_overview(training, runs, output):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))

    # 1. Do phu trong luc train
    ax = axes[0, 0]
    for label in ('QL-Original', 'QL-NewState'):
        rows = sorted((r for r in training if r['label'] == label), key=lambda r: int(r['episode']))
        eps = [int(r['episode']) for r in rows]
        cov = [float(r['coverage']) * 100 for r in rows]
        ax.plot(eps, cov, marker='o', linewidth=2, markersize=4.5, color=COLORS[label], label=label)
    ax.set_title('Training Coverage (Cologne1, 15 episodes)', fontsize=10.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Updated pairs / encountered pairs (%)')
    ax.set_xticks(range(1, 16, 2))
    ax.set_ylim(0, 65)
    ax.grid(alpha=0.25)
    ax.legend(frameon=True, fontsize=8.5)

    # 2. Ty le hoan thanh: Validation so voi Test
    ax = axes[0, 1]
    partitions = [('validation', 'Validation\n(seeds 42, 12345; n=4)'),
                  ('test', 'Held-out Test\n(seeds 41001-41003; n=6)')]
    width = 0.24
    for i, label in enumerate(VARIANTS):
        means, lows, highs = [], [], []
        for part_idx, (part, _) in enumerate(partitions):
            vals = [float(r['completion_rate']) * 100 for r in runs
                    if r['partition'] == part and r['label'] == label]
            m = statistics.mean(vals)
            means.append(m)
            lows.append(m - min(vals))
            highs.append(max(vals) - m)
            offsets = [0.0] if len(vals) == 1 else [0.08 * k / (len(vals) - 1) - 0.04 for k in range(len(vals))]
            x_pos = part_idx + (i - 1) * width
            ax.scatter([x_pos + off for off in offsets], vals, color='#1f1f1f', s=14, zorder=4)
        xs = [p + (i - 1) * width for p in range(len(partitions))]
        ax.bar(xs, means, yerr=[lows, highs], width=width, color=COLORS[label],
               capsize=4, label=label, error_kw={'lw': 1.2}, zorder=2)
        for x, m, h in zip(xs, means, highs):
            ax.text(x, m + h + 1.8, f'{m:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(range(len(partitions)), [p[1] for p in partitions])
    ax.set_ylabel('Valid completion rate (%)')
    ax.set_title('Valid Completion Rate: Validation vs Test', fontsize=10.5)
    ax.set_ylim(0, 116)
    ax.grid(axis='y', alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, fontsize=8, loc='upper center', ncol=3)

    # 3. Thoi gian cho tren tap Test
    _grouped_bar(axes[1, 0], runs, 'test', 'average_waiting_time',
                 'Held-out Test: Average Waiting Time (n=6)', 'seconds / valid vehicle')

    # 4. Khoang lau nhat mot movement khong co den xanh tren tap Test
    _grouped_bar(axes[1, 1], runs, 'test', 'longest_movement_no_green',
                 'Held-out Test: Longest Movement No-Green Gap (n=6)', 'seconds')

    fig.suptitle('Stage 8 (Cologne1): State 4-Bin & Guard-300 Overview', fontsize=13, fontweight='bold')
    fig.text(0.5, 0.01, 'Bars: mean; whiskers: observed min-max range; dots: individual runs.', ha='center', fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(output / 'overview.png', dpi=160)
    plt.close(fig)


def plot_training(training, output):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    specs = [
        ('coverage', 'Coverage (%)', 100.0, (0, 65)),
        ('average_queue_length', 'Average Queue Length (vehicles)', 1.0, (40, 120)),
        ('greedy_change_rate', 'Greedy Policy Change Rate (%)', 100.0, (0, 35)),
    ]
    for ax, (col, ylabel, scale, ylim) in zip(axes, specs):
        for label in ('QL-Original', 'QL-NewState'):
            rows = sorted((r for r in training if r['label'] == label), key=lambda r: int(r['episode']))
            eps = [int(r['episode']) for r in rows]
            vals = [float(r[col]) * scale for r in rows]
            ax.plot(eps, vals, marker='o', linewidth=2, markersize=4, color=COLORS[label], label=label)
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split(' (')[0], fontsize=10.5)
        ax.set_xticks(range(1, 16, 2))
        ax.set_ylim(ylim)
        ax.grid(alpha=0.25)
        ax.legend(frameon=True, fontsize=8.5)
    fig.suptitle('Stage 8 Training Dynamics (Cologne1, 15 episodes)', fontsize=12.5, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output / 'training_curves.png', dpi=160)
    plt.close(fig)


def plot_test_detail(runs, output):
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2))
    metrics = [
        ('completion_rate', 'Valid Completion Rate (%)', '%', 100.0, (0, 108)),
        ('average_waiting_time', 'Average Waiting Time', 's / valid vehicle', 1.0, None),
        ('average_queue_length', 'Average Queue Length', 'vehicles', 1.0, None),
        ('teleports', 'Teleport Events', 'events / hour', 1.0, None),
        ('longest_movement_no_green', 'Longest Movement No-Green Gap', 'seconds', 1.0, None),
        ('longest_same_action_green_seconds', 'Longest Same-Action Green Streak', 'seconds', 1.0, None),
    ]
    for ax, (metric, title, unit, scale, ylim) in zip(axes.flat, metrics):
        _grouped_bar(ax, runs, 'test', metric, title, unit, scale=scale, ylim=ylim)
    fig.suptitle('Stage 8 Held-out Test Comparison (Cologne1, seeds 41001–41003, n=6)', fontsize=13, fontweight='bold')
    fig.text(0.5, 0.01, 'Bars: mean; whiskers: min-max; dots: 6 individual runs (3 seeds x 2 repeats).', ha='center', fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(output / 'test_metrics.png', dpi=160)
    plt.close(fig)


def plot_seed_repeats(runs, output):
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    test_runs = [r for r in runs if r['partition'] == 'test']
    slots = [(41001, 1), (41001, 2), (41002, 1), (41002, 2), (41003, 1), (41003, 2)]
    width = 0.25
    for i, label in enumerate(VARIANTS):
        vals = []
        for seed, rep in slots:
            match = [r for r in test_runs if r['label'] == label and int(r['seed']) == seed and int(r['repeat']) == rep]
            vals.append(float(match[0]['completion_rate']) * 100)
        xs = [k + (i - 1) * width for k in range(len(slots))]
        ax.bar(xs, vals, width=width, color=COLORS[label], label=label)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.2, f'{v:.1f}%', ha='center', va='bottom', fontsize=7.5)
    ax.set_xticks(range(len(slots)), [f'seed {s}\nrepeat {r}' for s, r in slots])
    ax.set_ylabel('Valid completion rate (%)')
    ax.set_ylim(0, 112)
    ax.set_title('Stage 8 Held-out Test: Completion Rate per Seed & Repeat (Cologne1)', fontsize=11.5)
    ax.grid(axis='y', alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, fontsize=8.5, loc='upper center', ncol=3)
    fig.tight_layout()
    fig.savefig(output / 'test_seed_repeats.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage8', type=Path, default=Path('reports/stage8'))
    parser.add_argument('--output', type=Path, default=Path('reports/stage8_plots'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    training = read_csv(args.stage8 / 'training.csv')
    runs = read_csv(args.stage8 / 'runs.csv')
    plot_overview(training, runs, args.output)
    plot_training(training, args.output)
    plot_test_detail(runs, args.output)
    plot_seed_repeats(runs, args.output)
    print(f'Saved 4 charts to {args.output}')


if __name__ == '__main__':
    main()
