import os, sys, io, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8', errors='replace')
import numpy as np
from common import *
trapz = getattr(np, 'trapezoid', None) or np.trapz

gt = ground_truth()
folds = fold_map("new")
INHIVE = set(n for n in gt if IN_HIVE_RE.match(n))

SIZE = {}
def wh(n):
    if n not in SIZE:
        im = imread(imgpath(n)); SIZE[n] = (im.shape[1], im.shape[0])
    return SIZE[n]

def gtbox(n, cls):
    rows = [r for r in gt.get(n, []) if r[0] == cls]
    if not rows:
        return []
    W, H = wh(n)
    return [[(r[1]-r[3]/2)*W, (r[2]-r[4]/2)*H, (r[1]+r[3]/2)*W, (r[2]+r[4]/2)*H] for r in rows]

def iou1(a, B):
    B = np.asarray(B, float).reshape(-1, 4)
    x1 = np.maximum(a[0], B[:, 0]); y1 = np.maximum(a[1], B[:, 1])
    x2 = np.minimum(a[2], B[:, 2]); y2 = np.minimum(a[3], B[:, 3])
    inter = np.clip(x2-x1, 0, None)*np.clip(y2-y1, 0, None)
    aa = (a[2]-a[0])*(a[3]-a[1]); bb = (B[:, 2]-B[:, 0])*(B[:, 3]-B[:, 1])
    return inter/(aa+bb-inter+1e-9)

_GTC = {}
def ap_at(records, cls, pool, iou_t=0.5):
    preds = []; gtd = {}; ngt = 0
    for n in pool:
        k = (n, cls)
        if k not in _GTC:
            _GTC[k] = gtbox(n, cls)
        g = _GTC[k]
        if g:
            gtd[n] = g; ngt += len(g)
        for r in records.get(n, []):
            if r[0] == cls:
                preds.append((r[1], n, np.asarray(r[2:6], float)))
    if ngt == 0:
        return float('nan'), 0
    if not preds:
        return 0.0, ngt          # no predictions for this class -> AP = 0 by convention
    preds.sort(key=lambda x: -x[0])
    used = {n: np.zeros(len(v), bool) for n, v in gtd.items()}
    tp = np.zeros(len(preds)); fp = np.zeros(len(preds))
    for k2, (cf, n, box) in enumerate(preds):
        if n in gtd:
            io_ = iou1(box, gtd[n]); j = int(np.argmax(io_))
            if io_[j] >= iou_t and not used[n][j]:
                used[n][j] = True; tp[k2] = 1
            else:
                fp[k2] = 1
        else:
            fp[k2] = 1
    ctp = np.cumsum(tp); cfp = np.cumsum(fp)
    rec = ctp/ngt; prec = ctp/(ctp+cfp+1e-9)
    mrec = np.concatenate(([0.0], rec, [1.0])); mpre = np.concatenate(([1.0], prec, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(trapz(np.interp(x, mrec, mpre), x)), ngt

def ap5095(records, cls, pool):
    return float(np.nanmean([ap_at(records, cls, pool, t)[0] for t in np.arange(0.5, 1.0, 0.05)]))

def img_metrics(records, pool, thr):
    TP = FN = FP = TN = 0
    for n in pool:
        y = 1 if any(r[0] == 1 for r in gt.get(n, [])) else 0
        p = 1 if any(r[0] == 1 and r[1] >= thr for r in records.get(n, [])) else 0
        if y and p: TP += 1
        elif y and not p: FN += 1
        elif not y and p: FP += 1
        else: TN += 1
    rc = TP/(TP+FN) if TP+FN else float('nan')
    sp = TN/(TN+FP) if TN+FP else float('nan')
    pr = TP/(TP+FP) if TP+FP else float('nan')
    f1 = (2*pr*rc/(pr+rc)) if (TP+FP and TP+FN and (pr+rc) > 0) else 0.0
    return dict(TP=TP, FN=FN, FP=FP, TN=TN, acc=(TP+TN)/max(1, TP+TN+FP+FN),
                recall=rc, spec=sp, prec=pr, f1=f1)

R = {}
for t in ORDER:
    p = os.path.join(OOF, f"{t}.json")
    if os.path.exists(p):
        R[t] = json.load(open(p))
        print(f"loaded {t}: {len(R[t])} images")
print()

RESULTS = {}
for scope, poolset in (("FULL", set(gt)), ("INHIVE", INHIVE)):
    pool = [n for n in poolset if n in folds]
    npos = sum(1 for n in pool if any(r[0] == 1 for r in gt.get(n, [])))
    nmb = sum(r[0] == 1 for n in pool for r in gt.get(n, []))
    nb = sum(len(gt.get(n, [])) for n in pool)
    print("="*118)
    print(f"### {scope}: images={len(pool)}  mite-bearing images={npos} ({npos/len(pool)*100:.2f}%)  "
          f"mite boxes={nmb}/{nb} ({nmb/nb*100:.2f}%)")
    print("="*118)
    print(f"\n{'model':9s} {'mAP@.5':>8} {'mAP@.5:.95':>11} {'AP50 nonmite':>13} {'AP50 mite':>10} "
          f"{'mite AP50 per fold (mean+-SD)':>44}")
    RESULTS[scope] = {}
    for t in ORDER:
        if t not in R:
            continue
        a0, _ = ap_at(R[t], 0, pool); a1, _ = ap_at(R[t], 1, pool)
        m0 = ap5095(R[t], 0, pool); m1 = ap5095(R[t], 1, pool)
        v = []
        for i in range(1, 6):
            sub = [n for n in pool if folds.get(n) == i]
            v.append(ap_at(R[t], 1, sub)[0])
        v = np.array(v, float)
        fl = " ".join(f"{x:.3f}" for x in v) + f"  = {np.nanmean(v):.4f}+-{np.nanstd(v, ddof=1):.4f}"
        print(f"{t:9s} {np.nanmean([a0,a1]):>8.4f} {np.nanmean([m0,m1]):>11.4f} "
              f"{a0:>13.4f} {a1:>10.4f} {fl:>44}")
        RESULTS[scope][t] = dict(map50=float(np.nanmean([a0, a1])), map5095=float(np.nanmean([m0, m1])),
                                 ap50_nonmite=a0, ap50_mite=a1,
                                 fold_mite=[None if np.isnan(x) else x for x in v],
                                 mite_mean=float(np.nanmean(v)), mite_sd=float(np.nanstd(v, ddof=1)))
    print(f"\n{'model':9s} {'thr':>5} {'TP':>4} {'FN':>4} {'FP':>5} {'TN':>5} {'recall':>7} {'spec':>7} {'prec':>7} {'F1':>6}")
    for t in ORDER:
        if t not in R:
            continue
        for thr in (0.05, 0.25):
            m = img_metrics(R[t], pool, thr)
            print(f"{t:9s} {thr:>5} {m['TP']:>4} {m['FN']:>4} {m['FP']:>5} {m['TN']:>5} "
                  f"{m['recall']:>7.3f} {m['spec']:>7.3f} {m['prec']:>7.3f} {m['f1']:>6.3f}")
            RESULTS[scope].setdefault(t, {})[f"img{thr}"] = m
    print()

json.dump(RESULTS, open(os.path.join(SCR, "results_summary.json"), "w"), indent=1)
print("ANALYZE DONE -> results_summary.json")
