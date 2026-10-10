"""Extract best-epoch metrics from all folds for all models."""
import csv
import os
import re
import numpy as np

BATCH1 = os.path.join(os.environ.get("VARROA_RUNS", "runs"), "results_batch1")
V7V9 = os.path.join(os.environ.get("VARROA_RUNS", "runs"), "YOLO_V7_V9_V13_V26")

# Ultralytics models: results.csv
ultralytics_models = {
    'YOLOv8n':  os.path.join(BATCH1, 'yolov8n'),
    'YOLOv9c':  os.path.join(V7V9, 'yolov9c_精簡版', 'yolov9c'),
    'YOLOv10n': os.path.join(BATCH1, 'yolov10n'),
    'YOLO11n':  os.path.join(BATCH1, 'yolo11n'),
    'YOLO12n':  os.path.join(BATCH1, 'yolo12n'),
    'YOLOv13n': os.path.join(V7V9, 'yolov13n', 'yolov13n'),
    'YOLO26n':  os.path.join(V7V9, 'yolo26n', 'yolo26n'),
}

def parse_ultralytics(model_name, base_dir):
    fold_metrics = []
    for fold in range(1, 6):
        csv_path = os.path.join(base_dir, f'fold-{fold}', 'results.csv')
        if not os.path.exists(csv_path):
            print(f"  {model_name} fold-{fold}: MISSING {csv_path}")
            continue
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # Clean header names (may have leading spaces)
        if rows:
            cleaned = []
            for r in rows:
                cleaned.append({k.strip(): v for k, v in r.items()})
            rows = cleaned

        # Find best epoch by mAP50
        best = max(rows, key=lambda r: float(r.get('metrics/mAP50(B)', 0)))
        epoch = best.get('epoch', '?')
        prec = float(best.get('metrics/precision(B)', 0))
        rec = float(best.get('metrics/recall(B)', 0))
        map50 = float(best.get('metrics/mAP50(B)', 0))
        map5095 = float(best.get('metrics/mAP50-95(B)', 0))
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        fold_metrics.append({
            'fold': fold, 'epoch': epoch,
            'precision': prec, 'recall': rec, 'f1': f1,
            'mAP50': map50, 'mAP50-95': map5095
        })
    return fold_metrics

def parse_yolov7(base_dir):
    fold_metrics = []
    for fold in range(1, 6):
        txt_path = os.path.join(base_dir, f'fold-{fold}', 'results.txt')
        if not os.path.exists(txt_path):
            print(f"  YOLOv7 fold-{fold}: MISSING {txt_path}")
            continue
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        # YOLOv7 results.txt: columns are space-separated
        # epoch/total  gpu_mem  box  obj  cls  total  labels  img_size  P  R  mAP@.5  mAP@.5:.95  val_box  val_obj  val_cls
        best_map50 = -1
        best_row = None
        for line in lines:
            vals = line.strip().split()
            if len(vals) < 12:
                continue
            try:
                epoch_str = vals[0]  # e.g., "0/99"
                p = float(vals[8])
                r = float(vals[9])
                m50 = float(vals[10])
                m5095 = float(vals[11])
                if m50 > best_map50:
                    best_map50 = m50
                    best_row = {'epoch': epoch_str, 'precision': p, 'recall': r,
                                'mAP50': m50, 'mAP50-95': m5095}
            except (ValueError, IndexError):
                continue
        if best_row:
            p, r = best_row['precision'], best_row['recall']
            best_row['f1'] = 2*p*r/(p+r) if (p+r) > 0 else 0
            best_row['fold'] = fold
            fold_metrics.append(best_row)
    return fold_metrics

def summarize(name, metrics):
    if not metrics:
        print(f"\n{name}: No data")
        return
    print(f"\n{'='*80}")
    print(f"{name} — Fold-level results (best epoch by mAP@0.5)")
    print(f"{'='*80}")
    print(f"{'Fold':<6} {'Epoch':<8} {'Precision':<11} {'Recall':<11} {'F1':<11} {'mAP@0.5':<11} {'mAP@0.5:0.95':<13}")
    print('-'*80)
    for m in metrics:
        print(f"{m['fold']:<6} {str(m['epoch']):<8} {m['precision']:<11.4f} {m['recall']:<11.4f} {m['f1']:<11.4f} {m['mAP50']:<11.4f} {m['mAP50-95']:<13.4f}")

    # Mean ± SD
    for key in ['precision', 'recall', 'f1', 'mAP50', 'mAP50-95']:
        vals = [m[key] for m in metrics]
        mean = np.mean(vals)
        sd = np.std(vals, ddof=1) if len(vals) > 1 else 0
        print(f"  {key:<15}: {mean:.4f} ± {sd:.4f}")

if __name__ == '__main__':
    # YOLOv7
    v7_dir = os.path.join(V7V9, 'yolov7_精簡版', 'yolov7')
    v7_metrics = parse_yolov7(v7_dir)
    summarize('YOLOv7', v7_metrics)

    # Ultralytics models
    for name, base in ultralytics_models.items():
        metrics = parse_ultralytics(name, base)
        summarize(name, metrics)
