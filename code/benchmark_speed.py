import os
"""Benchmark inference speed for all loadable models on RTX 5070 Ti."""
import time
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

BATCH1 = Path(os.path.join(os.environ.get("VARROA_RUNS", "runs"), "results_batch1"))
V7V9 = Path(os.path.join(os.environ.get("VARROA_RUNS", "runs"), "YOLO_V7_V9_V13_V26"))

models = {
    'YOLOv8n':  BATCH1 / 'yolov8n' / 'fold-1' / 'weights' / 'best.pt',
    'YOLOv9c':  V7V9 / 'yolov9c_精簡版' / 'yolov9c' / 'fold-1' / 'weights' / 'best.pt',
    'YOLOv10n': BATCH1 / 'yolov10n' / 'fold-1' / 'weights' / 'best.pt',
    'YOLO11n':  BATCH1 / 'yolo11n' / 'fold-1' / 'weights' / 'best.pt',
    'YOLO12n':  BATCH1 / 'yolo12n' / 'fold-1' / 'weights' / 'best.pt',
    'YOLO26n':  V7V9 / 'yolo26n' / 'yolo26n' / 'fold-1' / 'weights' / 'best.pt',
}

dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
WARMUP = 10
RUNS = 50

def main():
    for name, pt_path in models.items():
        if not pt_path.exists():
            print(f'{name}: MISSING {pt_path}')
            continue
        try:
            m = YOLO(str(pt_path))
            # warmup
            for _ in range(WARMUP):
                m.predict(dummy_img, imgsz=640, verbose=False)
            torch.cuda.synchronize()

            times = []
            for _ in range(RUNS):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                m.predict(dummy_img, imgsz=640, verbose=False)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

            mean_t = np.mean(times)
            std_t = np.std(times)
            print(f'{name}: {mean_t:.1f} +/- {std_t:.1f} ms  (preprocess+inference+NMS, {RUNS} runs)')
            del m
            torch.cuda.empty_cache()
        except Exception as e:
            print(f'{name}: ERROR - {e}')

if __name__ == '__main__':
    main()
