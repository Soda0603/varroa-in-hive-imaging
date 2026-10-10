import os, sys, io, glob, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8', errors='replace')
REPO = os.environ.get("YOLOV7_REPO", "yolov7")
sys.path.insert(0, REPO)
os.chdir(REPO)
import torch, numpy as np, cv2
_o = torch.load
torch.load = lambda *a, **k: _o(*a, **{**k, 'weights_only': False})
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

BASE = os.environ.get("VARROA_DATA", "data")
GTDIR = os.path.join(BASE, "mydataset", "all")
OUT = os.environ.get("VARROA_OUT", "out")

# ---- ground truth (xywh normalised -> xyxy abs) ----
gt = {}
for f in glob.glob(os.path.join(GTDIR, "*.txt")):
    n = os.path.basename(f)[:-4]
    if n == "classes":
        continue
    rows = []
    for l in open(f, encoding='utf8', errors='replace'):
        p = l.split()
        if len(p) >= 5:
            rows.append([int(float(p[0]))] + [float(x) for x in p[1:5]])
    gt[n] = rows

# ---- 300-epoch folds: 5-Fold訓練/train-N -> exp(20+N) ----
folds = {}
for i in range(1, 6):
    for l in open(os.path.join(BASE, "5-Fold訓練", f"train-{i}", "val.txt"), encoding='utf8', errors='replace'):
        l = l.strip()
        if l:
            folds[os.path.splitext(os.path.basename(l))[0]] = i
print("300-epoch OOF images:", len(folds))

def imgpath(n):
    for ext in ('.jpg', '.jpeg', '.png', '.JPG'):
        p = os.path.join(GTDIR, n + ext)
        if os.path.exists(p):
            return p
    return None

dev = torch.device('cuda:0')
records = {}   # name -> list of [cls, conf, x1,y1,x2,y2]
for i in range(1, 6):
    wp = os.path.join(REPO, "runs", "train", f"exp{20+i}", "weights", "best.pt")
    model = attempt_load(wp, map_location=dev); model.eval()
    names = [n for n, f in folds.items() if f == i]
    done = 0
    with torch.no_grad():
        for n in names:
            fp = imgpath(n)
            if fp is None:
                continue
            im0 = cv2.imdecode(np.fromfile(fp, dtype=np.uint8), cv2.IMREAD_COLOR)
            if im0 is None:
                continue
            img = letterbox(im0, 640, stride=32, auto=True)[0][:, :, ::-1].transpose(2, 0, 1)
            t = torch.from_numpy(np.ascontiguousarray(img)).to(dev).float().div(255).unsqueeze(0)
            det = non_max_suppression(model(t)[0], 0.001, 0.45)[0]
            rows = []
            if det is not None and len(det):
                det[:, :4] = scale_coords(t.shape[2:], det[:, :4], im0.shape).round()
                for *xy, cf, cl in det.cpu().numpy():
                    rows.append([int(cl), float(cf), float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])])
            records[n] = rows
            done += 1
    print(f"  fold-{i} (exp{20+i}): {done} images inferred")
    del model; torch.cuda.empty_cache()

json.dump(records, open(os.path.join(OUT, "oof300_preds.json"), "w"))

# ---------- image-level ----------
print("\n=== 300-epoch YOLOv7, five-fold OUT-OF-FOLD, image-level (2,810 images) ===")
truth = {n: (1 if any(r[0] == 1 for r in gt.get(n, [])) else 0) for n in records}
npos = sum(truth.values())
print(f"ground-truth mite-bearing images: {npos} / {len(records)}  (prevalence {npos/len(records)*100:.2f}%)")
print(f"{'thr':>6} {'TP':>4} {'FN':>4} {'FP':>5} {'TN':>5} {'recall':>7} {'spec':>7} {'prec':>7} {'F1':>6}")
for thr in (0.01, 0.05, 0.10, 0.25, 0.50):
    TP = FN = FP = TN = 0
    for n, y in truth.items():
        pos = any(r[0] == 1 and r[1] >= thr for r in records[n])
        if y and pos: TP += 1
        elif y and not pos: FN += 1
        elif not y and pos: FP += 1
        else: TN += 1
    rc = TP/(TP+FN) if TP+FN else 0
    sp = TN/(TN+FP) if TN+FP else 0
    pr = TP/(TP+FP) if TP+FP else 0
    f1 = 2*pr*rc/(pr+rc) if pr+rc else 0
    print(f"{thr:>6} {TP:>4} {FN:>4} {FP:>5} {TN:>5} {rc:>7.3f} {sp:>7.3f} {pr:>7.3f} {f1:>6.3f}")

print("\n--- per-fold image-level recall at conf 0.05 ---")
for i in range(1, 6):
    ns = [n for n in records if folds.get(n) == i]
    P = [n for n in ns if truth[n] == 1]
    tp = sum(1 for n in P if any(r[0] == 1 and r[1] >= 0.05 for r in records[n]))
    fp = sum(1 for n in ns if truth[n] == 0 and any(r[0] == 1 and r[1] >= 0.05 for r in records[n]))
    print(f"  fold-{i} (exp{20+i}): GT mite imgs={len(P):3d}  TP={tp:3d}  recall={tp/len(P) if P else 0:.3f}  FP={fp:3d}/{len(ns)-len(P)}")

# ---------- box-level AP@0.5 ----------
def iou_mat(a, b):
    a = np.array(a, dtype=float).reshape(-1, 4); b = np.array(b, dtype=float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2-x1, 0, None) * np.clip(y2-y1, 0, None)
    aa = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]); bb = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)

def ap_class(cls, foldsel=None):
    preds = []; ngt = 0; gtd = {}
    for n in records:
        if foldsel and folds.get(n) != foldsel:
            continue
        h, w = None, None
        rows = [r for r in gt.get(n, []) if r[0] == cls]
        if rows:
            fp = imgpath(n)
            im = cv2.imdecode(np.fromfile(fp, dtype=np.uint8), cv2.IMREAD_COLOR)
            H, W = im.shape[:2]
            gtd[n] = [[(r[1]-r[3]/2)*W, (r[2]-r[4]/2)*H, (r[1]+r[3]/2)*W, (r[2]+r[4]/2)*H] for r in rows]
            ngt += len(rows)
        for r in records[n]:
            if r[0] == cls:
                preds.append((r[1], n, r[2:6]))
    if ngt == 0:
        return float('nan'), 0
    preds.sort(key=lambda x: -x[0])
    used = {n: np.zeros(len(v), bool) for n, v in gtd.items()}
    tp = np.zeros(len(preds)); fp = np.zeros(len(preds))
    for k, (cf, n, box) in enumerate(preds):
        if n in gtd:
            ious = iou_mat([box], gtd[n])[0]
            j = int(np.argmax(ious))
            if ious[j] >= 0.5 and not used[n][j]:
                used[n][j] = True; tp[k] = 1
            else:
                fp[k] = 1
        else:
            fp[k] = 1
    ctp = np.cumsum(tp); cfp = np.cumsum(fp)
    rec = ctp / ngt; prec = ctp / (ctp + cfp + 1e-9)
    mrec = np.concatenate(([0.0], rec, [1.0])); mpre = np.concatenate(([1.0], prec, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x)), ngt

print("\n=== 300-epoch YOLOv7, box-level AP@0.5 (out-of-fold, pooled over 2,810 images) ===")
for c, lab in ((0, "non-mite"), (1, "mite-bearing")):
    ap, ng = ap_class(c)
    print(f"  class {c} ({lab:12s}): AP@0.5 = {ap:.4f}   (GT boxes = {ng})")
print("\n--- per-fold mite AP@0.5 ---")
vals = []
for i in range(1, 6):
    ap, ng = ap_class(1, foldsel=i)
    vals.append(ap)
    print(f"  fold-{i} (exp{20+i}): mite AP@0.5 = {ap:.4f}  (GT mite boxes = {ng})")
v = np.array(vals, float)
print(f"  mean +/- SD = {np.nanmean(v):.4f} +/- {np.nanstd(v, ddof=1):.4f}")
