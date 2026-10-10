import os, sys, io, glob, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8', errors='replace')
import numpy as np, cv2
trapz = getattr(np, 'trapezoid', None) or np.trapz

BASE = os.environ.get("VARROA_DATA", "data")
GTDIR = os.path.join(BASE, "mydataset", "all")
OUT = os.environ.get("VARROA_OUT", "out")
records = json.load(open(os.path.join(OUT, "oof300_preds.json")))

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

folds = {}
for i in range(1, 6):
    for l in open(os.path.join(BASE, "5-Fold訓練", f"train-{i}", "val.txt"), encoding='utf8', errors='replace'):
        l = l.strip()
        if l:
            folds[os.path.splitext(os.path.basename(l))[0]] = i

def imgpath(n):
    for ext in ('.jpg', '.jpeg', '.png', '.JPG'):
        p = os.path.join(GTDIR, n + ext)
        if os.path.exists(p):
            return p

SIZE = {}
def wh(n):
    if n not in SIZE:
        im = cv2.imdecode(np.fromfile(imgpath(n), dtype=np.uint8), cv2.IMREAD_COLOR)
        SIZE[n] = (im.shape[1], im.shape[0])
    return SIZE[n]

def iou1(a, B):
    B = np.array(B, float).reshape(-1, 4)
    x1 = np.maximum(a[0], B[:, 0]); y1 = np.maximum(a[1], B[:, 1])
    x2 = np.minimum(a[2], B[:, 2]); y2 = np.minimum(a[3], B[:, 3])
    inter = np.clip(x2-x1, 0, None) * np.clip(y2-y1, 0, None)
    aa = (a[2]-a[0])*(a[3]-a[1]); bb = (B[:, 2]-B[:, 0])*(B[:, 3]-B[:, 1])
    return inter / (aa + bb - inter + 1e-9)

def ap_class(cls, foldsel=None):
    preds = []; gtd = {}; ngt = 0
    for n in records:
        if foldsel and folds.get(n) != foldsel:
            continue
        rows = [r for r in gt.get(n, []) if r[0] == cls]
        if rows:
            W, H = wh(n)
            gtd[n] = [[(r[1]-r[3]/2)*W, (r[2]-r[4]/2)*H, (r[1]+r[3]/2)*W, (r[2]+r[4]/2)*H] for r in rows]
            ngt += len(rows)
        for r in records[n]:
            if r[0] == cls:
                preds.append((r[1], n, np.array(r[2:6], float)))
    if ngt == 0:
        return float('nan'), 0, 0
    if not preds:
        return 0.0, ngt, 0
    preds.sort(key=lambda x: -x[0])
    used = {n: np.zeros(len(v), bool) for n, v in gtd.items()}
    tp = np.zeros(len(preds)); fp = np.zeros(len(preds))
    for k, (cf, n, box) in enumerate(preds):
        if n in gtd:
            io_ = iou1(box, gtd[n]); j = int(np.argmax(io_))
            if io_[j] >= 0.5 and not used[n][j]:
                used[n][j] = True; tp[k] = 1
            else:
                fp[k] = 1
        else:
            fp[k] = 1
    ctp = np.cumsum(tp); cfp = np.cumsum(fp)
    rec = ctp/ngt; prec = ctp/(ctp+cfp+1e-9)
    mrec = np.concatenate(([0.0], rec, [1.0])); mpre = np.concatenate(([1.0], prec, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(trapz(np.interp(x, mrec, mpre), x)), ngt, len(preds)

print("=== 300-epoch YOLOv7, box-level AP@0.5, out-of-fold, pooled over 2,810 images ===")
for c, lab in ((0, "non-mite"), (1, "mite-bearing")):
    ap, ng, npred = ap_class(c)
    print(f"  class {c} ({lab:12s}): AP@0.5 = {ap:.4f}   GT boxes = {ng}   predictions = {npred}")

print("\n--- per-fold AP@0.5 ---")
for c, lab in ((0, "non-mite"), (1, "mite")):
    vals = []
    print(f"  {lab}:")
    for i in range(1, 6):
        ap, ng, _ = ap_class(c, foldsel=i)
        vals.append(ap)
        print(f"    fold-{i} (exp{20+i}): AP@0.5 = {ap:.4f}  (GT boxes = {ng})")
    v = np.array(vals, float)
    print(f"    mean +/- SD = {np.nanmean(v):.4f} +/- {np.nanstd(v, ddof=1):.4f}")
