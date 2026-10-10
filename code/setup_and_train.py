#!/usr/bin/env python3
"""
Master training script: 8 YOLO models × 5-fold stratified CV × 300 epochs.
Models: YOLOv7, YOLOv8n, YOLOv9c, YOLOv10n, YOLO11n, YOLO12n, YOLOv13n, YOLO26n

Usage:
    python setup_and_train.py --setup          # Generate configs + download weights
    python setup_and_train.py --train all      # Train all models sequentially
    python setup_and_train.py --train yolov8n yolov9c  # Train specific models
    python setup_and_train.py --train-fold yolov8n 3   # Train specific fold
    python setup_and_train.py --status         # Show training progress
"""

import os
import sys
import time
import yaml
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# ─── Configuration ───────────────────────────────────────────────────
BASE_DIR    = Path(os.environ.get("VARROA_DATA", "data"))
FOLD_DIR    = BASE_DIR / '5-Fold_Stratified'
ALL_DIR     = BASE_DIR / 'mydataset' / 'all'
YOLOV7_DIR  = BASE_DIR / 'yolov7'
OUTPUT_DIR  = Path(os.environ.get("VARROA_OUT", "out"))
CONFIGS_DIR = OUTPUT_DIR / 'configs'
RUNS_DIR    = OUTPUT_DIR / 'runs'
WEIGHTS_DIR = OUTPUT_DIR / 'weights'
LOG_FILE    = OUTPUT_DIR / 'training_log.json'

EPOCHS     = 300
EPOCHS_PER_MODEL = {
    'yolov7': 100,
    'yolov8n': 100,
    'yolov9c': 100,
    'yolov10n': 100,
    'yolo11n': 100,
    'yolo12n': 100,
    'yolo13n': 100,
    'yolo26n': 100,
}
IMG_SIZE   = 640
WORKERS    = 2
NUM_FOLDS  = 5
SEED       = 42
DEVICE     = 0

NC = 2
CLASS_NAMES = {0: 'Bee without visible mite', 1: 'Bee with visible mite'}

# Batch sizes per model (optimized for RTX 5070 Ti 12GB VRAM)
# Nano models: batch=64 (7.4GB), Compact/Full: batch=16 (safe)
BATCH_SIZES = {
    'yolov7':   16,
    'yolov8n':  64,
    'yolov9c':  16,
    'yolov10n': 64,
    'yolo11n':  64,
    'yolo12n':  64,
    'yolo13n':  32,
    'yolo26n':  64,
}

# Model definitions - use absolute paths to pre-downloaded weights
MODELS = {
    'yolov7':   {'type': 'v7',          'weights_url': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt'},
    'yolov8n':  {'type': 'ultralytics', 'model_name': str(WEIGHTS_DIR / 'yolov8n.pt')},
    'yolov9c':  {'type': 'ultralytics', 'model_name': str(WEIGHTS_DIR / 'yolov9c.pt')},
    'yolov10n': {'type': 'ultralytics', 'model_name': str(WEIGHTS_DIR / 'yolov10n.pt')},
    'yolo11n':  {'type': 'ultralytics', 'model_name': str(WEIGHTS_DIR / 'yolo11n.pt')},
    'yolo12n':  {'type': 'ultralytics', 'model_name': str(WEIGHTS_DIR / 'yolo12n.pt')},
    'yolo13n':  {'type': 'v13',         'model_name': 'yolov13n.yaml'},
    'yolo26n':  {'type': 'ultralytics', 'model_name': str(WEIGHTS_DIR / 'yolo26n.pt')},
}


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f'[{ts}] {msg}')


# ─── Setup functions ─────────────────────────────────────────────────

def create_dirs():
    for d in [CONFIGS_DIR, RUNS_DIR, WEIGHTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    for model_name in MODELS:
        (RUNS_DIR / model_name).mkdir(exist_ok=True)
    log('Directories created.')


def create_fold_txt_absolute(fold_idx):
    """Create fold txt files with absolute paths for robust path resolution."""
    abs_dir = CONFIGS_DIR / f'fold-{fold_idx}'
    abs_dir.mkdir(exist_ok=True)

    for split in ['train', 'val']:
        src = FOLD_DIR / f'fold-{fold_idx}' / f'{split}.txt'
        dst = abs_dir / f'{split}.txt'
        with open(src, 'rb') as f:
            raw = f.read()
        for enc in ['utf-8', 'cp950', 'cp1252', 'latin-1']:
            try:
                text = raw.decode(enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        lines = text.splitlines()
        with open(dst, 'w', encoding='utf-8') as f:
            for line in lines:
                rel = line.strip()
                if not rel:
                    continue
                # ./mydataset/all/xxx.jpg -> absolute path
                clean = rel.replace('./', '').lstrip('/')
                abs_path = BASE_DIR / clean
                f.write(str(abs_path) + '\n')
    return abs_dir


def create_ultralytics_yaml(fold_idx):
    """Create data YAML for ultralytics models (v8/v9/v10/v11/v12/v26)."""
    fold_cfg_dir = CONFIGS_DIR / f'fold-{fold_idx}'
    data = {
        'path': str(BASE_DIR),
        'train': str(fold_cfg_dir / 'train.txt'),
        'val':   str(fold_cfg_dir / 'val.txt'),
        'nc': NC,
        'names': CLASS_NAMES,
    }
    yaml_path = CONFIGS_DIR / f'fold-{fold_idx}-ultralytics.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    return yaml_path


def create_yolov7_yaml(fold_idx):
    """Create data YAML for YOLOv7 format."""
    fold_cfg_dir = CONFIGS_DIR / f'fold-{fold_idx}'
    data = {
        'train': str(fold_cfg_dir / 'train.txt'),
        'val':   str(fold_cfg_dir / 'val.txt'),
        'nc': NC,
        'names': list(CLASS_NAMES.values()),
    }
    yaml_path = CONFIGS_DIR / f'fold-{fold_idx}-yolov7.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    return yaml_path


def create_all_configs():
    log('Creating fold configs with absolute paths...')
    for fold in range(1, NUM_FOLDS + 1):
        create_fold_txt_absolute(fold)
        create_ultralytics_yaml(fold)
        create_yolov7_yaml(fold)
        log(f'  Fold {fold} configs created.')
    log('All configs ready.')


def download_yolov7_weights():
    """Download YOLOv7 pretrained weights."""
    dst = WEIGHTS_DIR / 'yolov7.pt'
    if dst.exists():
        log(f'YOLOv7 weights already exist: {dst}')
        return dst
    url = MODELS['yolov7']['weights_url']
    log(f'Downloading YOLOv7 weights from {url}...')
    import urllib.request
    urllib.request.urlretrieve(url, str(dst))
    log(f'YOLOv7 weights saved: {dst} ({dst.stat().st_size / 1e6:.1f} MB)')
    return dst


def setup():
    """Full setup: create dirs, configs, download weights."""
    create_dirs()
    create_all_configs()
    download_yolov7_weights()
    log('\n=== Setup complete! Ready to train. ===')
    log(f'  Configs: {CONFIGS_DIR}')
    log(f'  Runs:    {RUNS_DIR}')
    log(f'  Weights: {WEIGHTS_DIR}')


# ─── Training functions ──────────────────────────────────────────────

def load_training_log():
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            return json.load(f)
    return {}


def save_training_log(log_data):
    with open(LOG_FILE, 'w') as f:
        json.dump(log_data, f, indent=2)


def is_fold_done(model_name, fold_idx):
    """Check if a fold training completed (not just has partial results)."""
    run_dir = RUNS_DIR / model_name / f'fold-{fold_idx}'
    done_marker = run_dir / '.done'
    if done_marker.exists():
        return True
    log_data = load_training_log()
    key = f'{model_name}_fold{fold_idx}'
    return log_data.get(key, {}).get('status') == 'done'


def mark_fold_done(model_name, fold_idx, elapsed_sec):
    run_dir = RUNS_DIR / model_name / f'fold-{fold_idx}'
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / '.done').write_text(f'completed at {datetime.now().isoformat()}\n')
    log_data = load_training_log()
    key = f'{model_name}_fold{fold_idx}'
    log_data[key] = {
        'status': 'done',
        'elapsed_sec': round(elapsed_sec, 1),
        'finished_at': datetime.now().isoformat(),
    }
    save_training_log(log_data)


def train_ultralytics_fold(model_name, model_file, fold_idx):
    """Train one fold using ultralytics API. Resumes from last.pt if target epochs match."""
    from ultralytics import YOLO

    run_dir = RUNS_DIR / model_name / f'fold-{fold_idx}'
    if is_fold_done(model_name, fold_idx):
        log(f'  SKIP {model_name} fold-{fold_idx} (already done)')
        return

    target_epochs = EPOCHS_PER_MODEL.get(model_name, EPOCHS)
    data_yaml = CONFIGS_DIR / f'fold-{fold_idx}-ultralytics.yaml'
    batch = BATCH_SIZES.get(model_name, 16)

    last_pt = run_dir / 'weights' / 'last.pt'
    if last_pt.exists():
        ckpt = YOLO(str(last_pt))
        ckpt_epochs = getattr(ckpt, 'ckpt', {}).get('epoch', 0) if hasattr(ckpt, 'ckpt') and ckpt.ckpt else 0
        ckpt_total = getattr(ckpt, 'ckpt', {}).get('train_args', {}).get('epochs', 0) if hasattr(ckpt, 'ckpt') and ckpt.ckpt else 0
        if ckpt_total == target_epochs and ckpt_epochs < target_epochs:
            log(f'  RESUME {model_name} fold-{fold_idx} from epoch {ckpt_epochs} | target={target_epochs}')
            t0 = time.time()
            ckpt.train(resume=True, workers=WORKERS)
            elapsed = time.time() - t0
            mark_fold_done(model_name, fold_idx, elapsed)
            log(f'  DONE {model_name} fold-{fold_idx} in {elapsed/60:.1f} min')
            return
        else:
            log(f'  Clearing stale checkpoint (was {ckpt_total} ep, now {target_epochs} ep)')
            import shutil
            if run_dir.exists():
                shutil.rmtree(run_dir)

    log(f'  Training {model_name} fold-{fold_idx} | batch={batch} | epochs={target_epochs}')
    t0 = time.time()

    model = YOLO(model_file)
    model.train(
        data=str(data_yaml),
        epochs=target_epochs,
        imgsz=IMG_SIZE,
        batch=batch,
        workers=WORKERS,
        device=DEVICE,
        project=str(RUNS_DIR / model_name),
        name=f'fold-{fold_idx}',
        exist_ok=True,
        seed=SEED,
        patience=50,
        save=True,
        save_period=50,
        plots=True,
        val=True,
        verbose=True,
    )

    elapsed = time.time() - t0
    mark_fold_done(model_name, fold_idx, elapsed)
    log(f'  DONE {model_name} fold-{fold_idx} in {elapsed/60:.1f} min')


def train_yolov7_fold(fold_idx):
    """Train one fold using YOLOv7's train.py."""
    model_name = 'yolov7'
    if is_fold_done(model_name, fold_idx):
        log(f'  SKIP {model_name} fold-{fold_idx} (already done)')
        return

    data_yaml = CONFIGS_DIR / f'fold-{fold_idx}-yolov7.yaml'
    weights = WEIGHTS_DIR / 'yolov7.pt'
    cfg = YOLOV7_DIR / 'cfg' / 'training' / 'myyolov7.yaml'
    hyp = YOLOV7_DIR / 'data' / 'hyp.scratch.custom.yaml'
    batch = BATCH_SIZES['yolov7']
    run_dir = RUNS_DIR / model_name

    log(f'  Training {model_name} fold-{fold_idx} | batch={batch} | epochs={EPOCHS}')
    t0 = time.time()

    cmd = [
        sys.executable, str(YOLOV7_DIR / 'train.py'),
        '--workers', str(WORKERS),
        '--device', str(DEVICE),
        '--batch-size', str(batch),
        '--data', str(data_yaml),
        '--img', str(IMG_SIZE), str(IMG_SIZE),
        '--cfg', str(cfg),
        '--weights', str(weights),
        '--name', f'fold-{fold_idx}',
        '--hyp', str(hyp),
        '--epochs', str(EPOCHS),
        '--project', str(run_dir),
        '--exist-ok',
    ]

    result = subprocess.run(cmd, cwd=str(YOLOV7_DIR))
    if result.returncode != 0:
        log(f'  ERROR {model_name} fold-{fold_idx} exited with code {result.returncode}')
        return

    elapsed = time.time() - t0
    mark_fold_done(model_name, fold_idx, elapsed)
    log(f'  DONE {model_name} fold-{fold_idx} in {elapsed/60:.1f} min')


def train_yolov13_fold(fold_idx):
    """Train one fold using YOLOv13. Requires: pip install git+https://github.com/iMoonLab/yolov13.git"""
    model_name = 'yolo13n'
    if is_fold_done(model_name, fold_idx):
        log(f'  SKIP {model_name} fold-{fold_idx} (already done)')
        return

    data_yaml = CONFIGS_DIR / f'fold-{fold_idx}-ultralytics.yaml'
    batch = BATCH_SIZES.get(model_name, 16)

    log(f'  Training {model_name} fold-{fold_idx} | batch={batch} | epochs={EPOCHS}')
    t0 = time.time()

    from ultralytics import YOLO
    model = YOLO('yolov13n.yaml')
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=batch,
        workers=WORKERS,
        device=DEVICE,
        project=str(RUNS_DIR / model_name),
        name=f'fold-{fold_idx}',
        exist_ok=True,
        seed=SEED,
        patience=50,
        save=True,
        save_period=50,
        plots=True,
        val=True,
        verbose=True,
    )

    elapsed = time.time() - t0
    mark_fold_done(model_name, fold_idx, elapsed)
    log(f'  DONE {model_name} fold-{fold_idx} in {elapsed/60:.1f} min')


def train_model(model_name, folds=None):
    """Train all folds for one model."""
    if folds is None:
        folds = list(range(1, NUM_FOLDS + 1))

    log(f'\n{"="*60}')
    log(f'Training {model_name} | folds={folds}')
    log(f'{"="*60}')

    info = MODELS.get(model_name)
    if info is None:
        log(f'Unknown model: {model_name}')
        log(f'Available: {", ".join(MODELS.keys())}')
        return

    t_model_start = time.time()

    for fold in folds:
        try:
            if info['type'] == 'v7':
                train_yolov7_fold(fold)
            elif info['type'] == 'v13':
                train_yolov13_fold(fold)
            else:
                train_ultralytics_fold(model_name, info['model_name'], fold)
        except Exception as e:
            log(f'  ERROR {model_name} fold-{fold}: {e}')
            log(f'  Continuing to next fold...')

    elapsed = time.time() - t_model_start
    log(f'\n{model_name} ALL FOLDS complete in {elapsed/60:.1f} min ({elapsed/3600:.1f} h)')


def train_all(model_names=None):
    """Train all specified models."""
    if model_names is None or model_names == ['all']:
        model_names = list(MODELS.keys())

    log(f'\n{"#"*60}')
    log(f'Training {len(model_names)} models × {NUM_FOLDS} folds × {EPOCHS} epochs')
    log(f'Models: {", ".join(model_names)}')
    log(f'Output: {RUNS_DIR}')
    log(f'{"#"*60}')

    t_total = time.time()
    for name in model_names:
        try:
            train_model(name)
        except Exception as e:
            log(f'ERROR training {name}: {e}')
            log(f'Continuing to next model...')
    elapsed = time.time() - t_total
    log(f'\n{"#"*60}')
    log(f'ALL TRAINING COMPLETE in {elapsed/3600:.1f} hours')
    log(f'{"#"*60}')


def show_status():
    """Show training progress."""
    log_data = load_training_log()
    print(f'\n{"Model":<12} {"F1":>4} {"F2":>4} {"F3":>4} {"F4":>4} {"F5":>4}  {"Total Time":>12}')
    print('-' * 58)
    for model_name in MODELS:
        times = []
        statuses = []
        for fold in range(1, NUM_FOLDS + 1):
            key = f'{model_name}_fold{fold}'
            entry = log_data.get(key, {})
            if entry.get('status') == 'done':
                statuses.append(' OK ')
                times.append(entry.get('elapsed_sec', 0))
            elif is_fold_done(model_name, fold):
                statuses.append(' OK ')
            else:
                statuses.append(' -- ')
        total_t = sum(times)
        t_str = f'{total_t/3600:.1f}h' if total_t > 0 else '--'
        print(f'{model_name:<12} {"".join(statuses)}  {t_str:>12}')
    print()


# ─── CLI ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description='YOLO 5-Fold CV Training')
    parser.add_argument('--setup', action='store_true', help='Create configs and download weights')
    parser.add_argument('--train', nargs='*', help='Train models (names or "all")')
    parser.add_argument('--train-fold', nargs=2, metavar=('MODEL', 'FOLD'), help='Train specific model and fold')
    parser.add_argument('--status', action='store_true', help='Show training progress')
    parser.add_argument('--batch', type=int, help='Override batch size')
    parser.add_argument('--epochs', type=int, help='Override epochs')

    args = parser.parse_args()

    if args.epochs:
        EPOCHS = args.epochs
    if args.batch:
        for k in BATCH_SIZES:
            BATCH_SIZES[k] = args.batch

    if args.setup:
        setup()
    elif args.train is not None:
        names = args.train if args.train else ['all']
        train_all(names)
    elif args.train_fold:
        model_name, fold = args.train_fold
        train_model(model_name, folds=[int(fold)])
    elif args.status:
        show_status()
    else:
        parser.print_help()
        print('\nQuick start:')
        print('  1. python setup_and_train.py --setup')
        print('  2. python setup_and_train.py --train all')
        print('  3. python setup_and_train.py --status')
