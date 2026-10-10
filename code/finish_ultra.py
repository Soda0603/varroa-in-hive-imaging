"""Finish the OOF prediction set:
   (1) YOLO26n full run, (2) YOLOv13n imported from the authors' own predictions.jsonl
   (the iMoonLab fork needed to load its weights is not available locally),
   (3) backfill the 11 cp950-named LINE_ALBUM images for every loadable ultralytics model."""
import os, sys, io, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8', errors='replace')
from common import *
from ultralytics import YOLO

folds = fold_map("new")
print("OOF images:", len(folds), flush=True)

# ---------- (1) YOLO26n ----------
dst = os.path.join(OOF, "YOLO26n.json")
if not os.path.exists(dst):
    t0 = time.time(); rec = {}
    for i in range(1, 6):
        m = YOLO(os.path.join(MW, MODEL_DIRS["YOLO26n"], f"fold-{i}", "best.pt"))
        for n in [x for x, f in folds.items() if f == i]:
            fp = imgpath(n)
            if fp is None:
                continue
            r = m.predict(source=imread(fp), conf=0.001, iou=0.45, imgsz=640, verbose=False, device=0)[0]
            b = r.boxes; rows = []
            if len(b):
                xy = b.xyxy.cpu().numpy(); cf = b.conf.cpu().numpy(); cl = b.cls.cpu().numpy()
                rows = [[int(cl[k]), round(float(cf[k]), 5), round(float(xy[k][0]), 1),
                         round(float(xy[k][1]), 1), round(float(xy[k][2]), 1), round(float(xy[k][3]), 1)]
                        for k in range(len(b))]
            rec[n] = rows
        print(f"  YOLO26n fold-{i} done", flush=True)
        del m
    json.dump(rec, open(dst, "w"))
    print(f"YOLO26n: {len(rec)} images, {time.time()-t0:.0f}s", flush=True)

# ---------- (2) YOLOv13n from authors' predictions.jsonl ----------
dst = os.path.join(OOF, "YOLOv13n.json")
if not os.path.exists(dst):
    SRC = os.path.join(os.environ.get("VARROA_RUNS", "runs"), "yolov13n")
    rec = {}
    for i in range(1, 6):
        for line in open(os.path.join(SRC, f"fold-{i}", "predictions.jsonl"), encoding='utf8', errors='replace'):
            r = json.loads(line)
            n = os.path.splitext(r["image"])[0]
            rec[n] = [[int(b[5]), round(float(b[4]), 5), round(float(b[0]), 1), round(float(b[1]), 1),
                       round(float(b[2]), 1), round(float(b[3]), 1)] for b in r["predictions"]]
    json.dump(rec, open(dst, "w"))
    print(f"YOLOv13n: imported {len(rec)} images from authors' predictions.jsonl", flush=True)

# ---------- (3) backfill the 11 unresolved images ----------
LOADABLE = ["YOLOv8n", "YOLOv9c", "YOLOv10n", "YOLO11n", "YOLO12n", "YOLO26n"]
for tag in LOADABLE:
    p = os.path.join(OOF, f"{tag}.json")
    if not os.path.exists(p):
        print(f"{tag}: absent, skip"); continue
    rec = json.load(open(p))
    todo = [n for n in folds if n not in rec]
    if not todo:
        print(f"{tag}: complete ({len(rec)})", flush=True); continue
    byfold = {}
    for n in todo:
        byfold.setdefault(folds[n], []).append(n)
    for i, names in sorted(byfold.items()):
        m = YOLO(os.path.join(MW, MODEL_DIRS[tag], f"fold-{i}", "best.pt"))
        for n in names:
            fp = imgpath(n)
            if fp is None:
                print(f"  !! {tag}: image not found {n!r}"); continue
            r = m.predict(source=imread(fp), conf=0.001, iou=0.45, imgsz=640, verbose=False, device=0)[0]
            b = r.boxes; rows = []
            if len(b):
                xy = b.xyxy.cpu().numpy(); cf = b.conf.cpu().numpy(); cl = b.cls.cpu().numpy()
                rows = [[int(cl[k]), round(float(cf[k]), 5), round(float(xy[k][0]), 1),
                         round(float(xy[k][1]), 1), round(float(xy[k][2]), 1), round(float(xy[k][3]), 1)]
                        for k in range(len(b))]
            rec[n] = rows
        del m
    json.dump(rec, open(p, "w"))
    print(f"{tag}: backfilled {len(todo)} -> {len(rec)}", flush=True)
print("FINISH_ULTRA DONE", flush=True)
