import os, sys, io, json, time, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8', errors='replace')
import numpy as np, cv2
from ultralytics import YOLO

BASE = os.environ.get("VARROA_DATA", "data")
MW = os.environ.get("VARROA_WEIGHTS", "weights")
ALLD = os.path.join(BASE, "mydataset", "all")
OUT = os.path.join(os.environ.get("VARROA_OUT", "out"), "oof2823")
os.makedirs(OUT, exist_ok=True)

folds = {}
for i in range(1, 6):
    for l in open(os.path.join(BASE, "5-Fold_Stratified", f"fold-{i}", "val.txt"), encoding='utf8', errors='replace'):
        l = l.strip()
        if l:
            folds[os.path.splitext(os.path.basename(l))[0]] = i
print("OOF images:", len(folds), flush=True)

def imgpath(n):
    for ext in ('.jpg', '.jpeg', '.png', '.JPG'):
        p = os.path.join(ALLD, n + ext)
        if os.path.exists(p):
            return p

MODELS = {
    "YOLOv8n":  "yolov8n_5fold_weights",
    "YOLOv9c":  "yolov9c_5fold_weights",
    "YOLOv10n": "yolov10n_5fold_weights",
    "YOLO11n":  "yolo11n_5fold_weights",
    "YOLO12n":  "yolo12n_5fold_weights",
    "YOLOv13n": "yolov13n_5fold_weights",
    "YOLO26n":  "yolo26n_5fold_weights",
}

for tag, folder in MODELS.items():
    dst = os.path.join(OUT, f"{tag}.json")
    if os.path.exists(dst):
        print(f"{tag}: already done, skip", flush=True)
        continue
    t0 = time.time()
    rec = {}
    for i in range(1, 6):
        wp = os.path.join(MW, folder, f"fold-{i}", "best.pt")
        if not os.path.exists(wp):
            print(f"  !! {tag} fold-{i} weight missing", flush=True); continue
        m = YOLO(wp)
        names = [n for n, f in folds.items() if f == i]
        for n in names:
            fp = imgpath(n)
            if fp is None:
                continue
            im = cv2.imdecode(np.fromfile(fp, dtype=np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                continue
            r = m.predict(source=im, conf=0.001, iou=0.45, imgsz=640, verbose=False, device=0)[0]
            b = r.boxes
            rows = []
            if len(b):
                xy = b.xyxy.cpu().numpy(); cf = b.conf.cpu().numpy(); cl = b.cls.cpu().numpy()
                for k in range(len(b)):
                    rows.append([int(cl[k]), round(float(cf[k]), 5),
                                 round(float(xy[k][0]), 1), round(float(xy[k][1]), 1),
                                 round(float(xy[k][2]), 1), round(float(xy[k][3]), 1)])
            rec[n] = rows
        print(f"  {tag} fold-{i}: {len(names)} images", flush=True)
        del m
    json.dump(rec, open(dst, "w"))
    print(f"{tag}: {len(rec)} images, {time.time()-t0:.0f}s -> {dst}", flush=True)
print("ULTRA DONE", flush=True)
