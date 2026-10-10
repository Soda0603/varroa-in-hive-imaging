#!/usr/bin/env python3
"""
Collect all 5-fold CV results and generate comparison tables for the manuscript.
Run after all training is complete.

Usage:
    python analyze_results.py              # Generate full comparison
    python analyze_results.py --latex      # Output LaTeX table
    python analyze_results.py --csv        # Output CSV
"""

import os
import json
import argparse
import csv
from pathlib import Path
from collections import defaultdict

import yaml

RUNS_DIR = Path(os.environ.get("VARROA_RUNS", "runs"))
OUTPUT_DIR = Path(os.path.join(os.environ.get("VARROA_OUT", "out"), "analysis"))

MODELS = ['yolov7', 'yolov8n', 'yolov9c', 'yolov10n', 'yolo11n', 'yolo12n', 'yolo13n', 'yolo26n']
NUM_FOLDS = 5

METRICS = ['precision', 'recall', 'mAP50', 'mAP50-95']
METRIC_KEYS_ULTRALYTICS = {
    'precision':  'metrics/precision(B)',
    'recall':     'metrics/recall(B)',
    'mAP50':      'metrics/mAP50(B)',
    'mAP50-95':   'metrics/mAP50-95(B)',
}


def read_ultralytics_results(run_dir):
    """Read results from ultralytics training run."""
    results_csv = run_dir / 'results.csv'
    if not results_csv.exists():
        return None

    with open(results_csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    best_row = max(rows, key=lambda r: float(r.get('metrics/mAP50(B)', r.get('      metrics/mAP50(B)', '0')).strip()))

    result = {}
    for metric, key in METRIC_KEYS_ULTRALYTICS.items():
        for k in [key, '      ' + key.split('/')[-1], '   ' + key]:
            if k in best_row:
                result[metric] = float(best_row[k].strip())
                break
            clean_keys = {kk.strip(): kk for kk in best_row.keys()}
            if key in clean_keys:
                result[metric] = float(best_row[clean_keys[key]].strip())
                break

    if not result:
        for k, v in best_row.items():
            k_clean = k.strip()
            if 'precision' in k_clean.lower():
                result['precision'] = float(v.strip())
            elif 'recall' in k_clean.lower() and 'mAP' not in k_clean:
                result['recall'] = float(v.strip())
            elif 'mAP50-95' in k_clean or 'mAP50_95' in k_clean:
                result['mAP50-95'] = float(v.strip())
            elif 'mAP50' in k_clean and '95' not in k_clean:
                result['mAP50'] = float(v.strip())

    epochs_trained = len(rows)
    result['epochs'] = epochs_trained

    return result


def read_yolov7_results(run_dir):
    """Read results from YOLOv7 training run."""
    results_txt = run_dir / 'results.txt'
    if not results_txt.exists():
        return None

    best_map50 = 0
    best_row = None
    with open(results_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                try:
                    p, r, map50, map5095 = float(parts[8]), float(parts[9]), float(parts[10]), float(parts[11])
                    if map50 > best_map50:
                        best_map50 = map50
                        best_row = {'precision': p, 'recall': r, 'mAP50': map50, 'mAP50-95': map5095}
                except (ValueError, IndexError):
                    continue

    if best_row:
        best_row['epochs'] = sum(1 for _ in open(results_txt))
    return best_row


def collect_all_results():
    """Collect results from all models and folds."""
    all_results = {}

    for model_name in MODELS:
        model_dir = RUNS_DIR / model_name
        if not model_dir.exists():
            continue

        fold_results = []
        for fold in range(1, NUM_FOLDS + 1):
            fold_dir = model_dir / f'fold-{fold}'
            if not fold_dir.exists():
                fold_results.append(None)
                continue

            if model_name == 'yolov7':
                result = read_yolov7_results(fold_dir)
            else:
                result = read_ultralytics_results(fold_dir)

            fold_results.append(result)

        all_results[model_name] = fold_results

    return all_results


def compute_statistics(all_results):
    """Compute mean and std for each metric across folds."""
    import statistics

    stats = {}
    for model_name, fold_results in all_results.items():
        valid = [r for r in fold_results if r is not None]
        if not valid:
            continue

        model_stats = {}
        for metric in METRICS:
            values = [r.get(metric, 0) for r in valid if metric in r]
            if values:
                model_stats[metric] = {
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values),
                    'n': len(values),
                    'values': values,
                }
        model_stats['folds_complete'] = len(valid)
        model_stats['folds_total'] = NUM_FOLDS
        stats[model_name] = model_stats

    return stats


def print_comparison_table(stats):
    """Print formatted comparison table."""
    print('\n' + '=' * 90)
    print('5-Fold Cross-Validation Results (Mean ± Std)')
    print('=' * 90)
    header = f'{"Model":<12} {"Precision":>14} {"Recall":>14} {"mAP50":>14} {"mAP50-95":>14} {"Folds":>6}'
    print(header)
    print('-' * 90)

    for model_name in MODELS:
        if model_name not in stats:
            print(f'{model_name:<12} {"--":>14} {"--":>14} {"--":>14} {"--":>14} {"0/5":>6}')
            continue

        s = stats[model_name]
        folds = f'{s["folds_complete"]}/{s["folds_total"]}'
        cols = []
        for metric in METRICS:
            if metric in s:
                m = s[metric]
                cols.append(f'{m["mean"]:.4f}±{m["std"]:.4f}')
            else:
                cols.append('--')
        print(f'{model_name:<12} {cols[0]:>14} {cols[1]:>14} {cols[2]:>14} {cols[3]:>14} {folds:>6}')

    print('=' * 90)


def print_per_fold_table(all_results):
    """Print per-fold results."""
    for metric in METRICS:
        print(f'\n--- {metric} per fold ---')
        header = f'{"Model":<12}' + ''.join([f'  Fold-{i}' for i in range(1, NUM_FOLDS + 1)]) + '    Mean'
        print(header)
        print('-' * 70)
        for model_name in MODELS:
            if model_name not in all_results:
                continue
            fold_results = all_results[model_name]
            vals = []
            for r in fold_results:
                if r and metric in r:
                    vals.append(r[metric])
                    print(f'{r[metric]:>8.4f}', end='')
                else:
                    print(f'{"--":>8}', end='')
            if vals:
                import statistics
                print(f'  {statistics.mean(vals):>8.4f}', end='')
            print(f'  {model_name}')


def generate_latex_table(stats):
    """Generate LaTeX table for the manuscript."""
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{5-Fold Cross-Validation Results for Varroa Mite Detection}')
    lines.append(r'\label{tab:model_comparison}')
    lines.append(r'\begin{tabular}{lcccc}')
    lines.append(r'\hline')
    lines.append(r'Model & Precision & Recall & mAP@50 & mAP@50-95 \\')
    lines.append(r'\hline')

    best = {m: 0 for m in METRICS}
    best_model = {m: '' for m in METRICS}
    for model_name, s in stats.items():
        for metric in METRICS:
            if metric in s and s[metric]['mean'] > best[metric]:
                best[metric] = s[metric]['mean']
                best_model[metric] = model_name

    for model_name in MODELS:
        if model_name not in stats:
            continue
        s = stats[model_name]
        cols = []
        for metric in METRICS:
            if metric in s:
                m = s[metric]
                val = f'{m["mean"]:.4f} $\\pm$ {m["std"]:.4f}'
                if best_model[metric] == model_name:
                    val = r'\textbf{' + val + '}'
                cols.append(val)
            else:
                cols.append('--')
        lines.append(f'{model_name} & {" & ".join(cols)} \\\\')

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


def generate_csv(stats, output_path):
    """Generate CSV output."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Metric', 'Mean', 'Std', 'Min', 'Max', 'N_Folds'])
        for model_name in MODELS:
            if model_name not in stats:
                continue
            s = stats[model_name]
            for metric in METRICS:
                if metric in s:
                    m = s[metric]
                    writer.writerow([model_name, metric, f'{m["mean"]:.6f}', f'{m["std"]:.6f}',
                                   f'{m["min"]:.6f}', f'{m["max"]:.6f}', m['n']])
    print(f'CSV saved to {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Analyze 5-fold CV results')
    parser.add_argument('--latex', action='store_true', help='Generate LaTeX table')
    parser.add_argument('--csv', action='store_true', help='Generate CSV output')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    print('Collecting results from all models...')
    all_results = collect_all_results()
    stats = compute_statistics(all_results)

    print_comparison_table(stats)
    print_per_fold_table(all_results)

    if args.latex:
        latex = generate_latex_table(stats)
        latex_path = OUTPUT_DIR / 'comparison_table.tex'
        with open(latex_path, 'w') as f:
            f.write(latex)
        print(f'\nLaTeX table saved to {latex_path}')
        print(latex)

    if args.csv:
        csv_path = OUTPUT_DIR / 'results_summary.csv'
        generate_csv(stats, csv_path)

    if not args.latex and not args.csv:
        latex = generate_latex_table(stats)
        latex_path = OUTPUT_DIR / 'comparison_table.tex'
        with open(latex_path, 'w') as f:
            f.write(latex)
        csv_path = OUTPUT_DIR / 'results_summary.csv'
        generate_csv(stats, csv_path)
        print(f'\nAll outputs saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
