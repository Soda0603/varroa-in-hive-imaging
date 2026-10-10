# -*- coding: utf-8 -*-
"""Figure 10: out-of-fold precision-recall curves, by class and by evaluation scope.

Design: 8 architectures are drawn as a recessive grey ensemble (their identity is
carried by Tables 4-7, not by colour) with the best-performing architecture of each
panel highlighted in a single hue and direct-labelled. Rationale: an 8-hue
categorical palette cannot clear the all-pairs CVD/normal-vision separation floors
for overlapping line series (validated in check_palette.py), so the series are cut
rather than coloured."""
import os, sys, io, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8', errors='replace')
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from common import *

OUTDIR = os.environ.get("VARROA_OUT", "out")
HILITE = "#2a78d6"     # categorical slot 1
INK = "#0b0b0b"
INK2 = "#52514e"
GREY = "#9c9b94"
GRID = "#e8e7e2"
SURF = "#ffffff"

avail = {f.name for f in font_manager.fontManager.ttflist}
for cand in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"):
    if cand in avail:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.labelsize": 8.5, "axes.labelcolor": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "savefig.facecolor": SURF,
})

gt = ground_truth()
folds = fold_map("new")
INHIVE = set(n for n in gt if IN_HIVE_RE.match(n))

import cv2
SIZE = {}
def wh(n):
    if n not in SIZE:
        im = imread(imgpath(n)); SIZE[n] = (im.shape[1], im.shape[0])
    return SIZE[n]

_GTC = {}
def gtbox(n, cls):
    k = (n, cls)
    if k not in _GTC:
        rows = [r for r in gt.get(n, []) if r[0] == cls]
        if not rows:
            _GTC[k] = []
        else:
            W, H = wh(n)
            _GTC[k] = [[(r[1]-r[3]/2)*W, (r[2]-r[4]/2)*H,
                        (r[1]+r[3]/2)*W, (r[2]+r[4]/2)*H] for r in rows]
    return _GTC[k]

def iou1(a, B):
    B = np.asarray(B, float).reshape(-1, 4)
    x1 = np.maximum(a[0], B[:, 0]); y1 = np.maximum(a[1], B[:, 1])
    x2 = np.minimum(a[2], B[:, 2]); y2 = np.minimum(a[3], B[:, 3])
    inter = np.clip(x2-x1, 0, None)*np.clip(y2-y1, 0, None)
    aa = (a[2]-a[0])*(a[3]-a[1]); bb = (B[:, 2]-B[:, 0])*(B[:, 3]-B[:, 1])
    return inter/(aa+bb-inter+1e-9)

def pr_curve(records, cls, pool):
    preds = []; gtd = {}; ngt = 0
    for n in pool:
        g = gtbox(n, cls)
        if g:
            gtd[n] = g; ngt += len(g)
        for r in records.get(n, []):
            if r[0] == cls:
                preds.append((r[1], n, np.asarray(r[2:6], float)))
    if ngt == 0 or not preds:
        return np.array([0.0]), np.array([0.0]), ngt, 0.0
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
    rec = ctp/ngt; prec = ctp/(ctp+cfp)
    # AP (101-point, for picking the highlighted architecture)
    mrec = np.concatenate(([0.0], rec, [1.0])); mpre = np.concatenate(([1.0], prec, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trapz = getattr(np, 'trapezoid', None) or np.trapz
    ap = float(trapz(np.interp(x, mrec, mpre), x))
    return rec, prec, ngt, ap

R = {}
for t in ORDER:
    p = os.path.join(OOF, f"{t}.json")
    if os.path.exists(p):
        R[t] = json.load(open(p))
print("models:", list(R), flush=True)

SCOPES = [("Full dataset (2,823 images)", [n for n in gt if n in folds]),
          ("In-hive frames only (2,795 images)", [n for n in INHIVE if n in folds])]
CLASSES = [(0, "Bee without visible mite"), (1, "Bee with visible mite")]

curves = {}
for si, (sname, pool) in enumerate(SCOPES):
    for cls, cname in CLASSES:
        for t in R:
            curves[(si, cls, t)] = pr_curve(R[t], cls, pool)
            print(f"  computed {sname[:12]} cls{cls} {t}", flush=True)

fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.1), constrained_layout=True)
letters = [["(a)", "(b)"], ["(c)", "(d)"]]
k = 0
for ci, (cls, cname) in enumerate(CLASSES):
    for si, (sname, pool) in enumerate(SCOPES):
        ax = axes[ci][si]
        aps = {t: curves[(si, cls, t)][3] for t in R}
        best = max(aps, key=aps.get)
        for t in R:
            rec, prec, ngt, ap = curves[(si, cls, t)]
            if t == best:
                continue
            ax.step(rec, prec, where='post', color=GREY, lw=0.8, alpha=0.85,
                    solid_capstyle='butt', zorder=2)
        rec, prec, ngt, ap = curves[(si, cls, best)]
        ax.step(rec, prec, where='post', color=HILITE, lw=1.8,
                solid_capstyle='butt', zorder=3)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(True, color=GRID, lw=0.4, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if ci == 1:
            ax.set_xlabel("Recall")
        if si == 0:
            ax.set_ylabel("Precision")
            ax.text(-0.235, 0.5, cname, transform=ax.transAxes, rotation=90,
                    va='center', ha='center', fontsize=8.5, color=INK, fontweight='bold')
        ax.set_title(letters[ci][si], loc='left', fontsize=9.5,
                     fontweight='bold', color=INK, pad=7)
        if ci == 0:
            ax.set_title(sname, loc='center', fontsize=9.0,
                         fontweight='bold', color=INK, pad=7)
        # label the highlighted curve directly; blue text matches the blue line, no leader
        lbl = (f"{best}\nAP@0.5 = {ap:.3f}" if ap > 0.02
               else f"highest AP@0.5 = {ap:.4f} ({best})")
        if cls == 0:
            ax.text(0.97, 0.90, lbl, transform=ax.transAxes, fontsize=7.5,
                    color=HILITE, ha='right', va='top', linespacing=1.4)
        else:
            ax.text(0.5, 0.60, lbl, transform=ax.transAxes, fontsize=7.5,
                    color=HILITE, ha='center', va='center', linespacing=1.4)
        nz = sum(1 for t in R if curves[(si, cls, t)][3] <= 0.0)
        note = f"{ngt:,} ground-truth boxes"
        if cls == 1:
            note += f"\nall eight AP@0.5 \u2264 {max(aps.values()):.3f}"
            if nz:
                note += f"\n{nz} of 8 produce no true positive at IoU \u2265 0.5"
        ax.text(0.97, 0.04, note, transform=ax.transAxes, fontsize=7.0,
                color=INK2, ha='right', va='bottom', linespacing=1.4)
        k += 1

from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color=GREY, lw=0.9,
                           label="Individual architectures (n = 8; see Tables 4\u20137)"),
                    Line2D([], [], color=HILITE, lw=1.8,
                           label="Highest-AP architecture in panel")],
           loc='outside lower center', ncols=2, frameon=False, fontsize=7.8,
           handlelength=2.4, labelcolor=INK2)

png = os.path.join(OUTDIR, "Figure10_PR_curves.png")
pdf = os.path.join(OUTDIR, "Figure10_PR_curves.pdf")
fig.savefig(png, dpi=600)
fig.savefig(pdf)
print("saved:", png)
print("saved:", pdf)

print("\n=== AP@0.5 used for highlighting ===")
for ci, (cls, cname) in enumerate(CLASSES):
    for si, (sname, pool) in enumerate(SCOPES):
        aps = {t: round(curves[(si, cls, t)][3], 4) for t in R}
        print(f"  cls{cls} {sname[:22]:22s} {aps}")
