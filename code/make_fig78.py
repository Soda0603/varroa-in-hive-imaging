# -*- coding: utf-8 -*-
"""Regenerate Figures 7 and 8 from out-of-fold predictions with the current class names."""
import os, sys, io, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8', errors='replace')
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from common import *

MODEL = "YOLO11n"
CONF = 0.25          # display threshold for predicted boxes
OUT = os.environ.get("VARROA_OUT", "out")

BLUE = (42, 120, 214)     # predicted box  (categorical slot 1)
RED = (227, 73, 72)       # ground-truth mite-bearing (slot 8)
INK = (11, 11, 11)
INK2 = (82, 81, 78)
WHITE = (255, 255, 255)

FIG7 = "img-20240116_144310--2496"
FIG8_FN = ["img-20240307_051133--1913", "img-20240307_051427--1921", "img-20240307_051805--1931"]
FIG8_TN = ["img-20240305_110446--513", "img-20240305_104421--177", "img-20240116_123707--429"]

gt = ground_truth()
folds = fold_map("new")
R = json.load(open(os.path.join(OOF, f"{MODEL}.json")))

# mite-level boxes annotated manually for this figure (LabelImg, class 15 = "Mite")
MITE_DIR = os.environ.get("VARROA_MITE_IMAGES", "mite_images")
def read_mite(n, W, H):
    p = os.path.join(MITE_DIR, n + ".txt")
    if not os.path.exists(p):
        return []
    out = []
    for l in open(p, encoding='utf8', errors='replace'):
        q = l.split()
        if len(q) >= 5:
            cx, cy, w, h = (float(v) for v in q[1:5])
            out.append([(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H])
    return out

def F(sz, bold=False):
    for p in ([r"C:\Windows\Fonts\arialbd.ttf"] if bold else [r"C:\Windows\Fonts\arial.ttf"]):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

def stamp(n):
    m = re.match(r'^img-(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})--', n)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)}:{m.group(6)}" if m else n

def load(n):
    im = imread(imgpath(n))
    return Image.fromarray(im[:, :, ::-1])

def gtbox(n, cls, W, H):
    return [[(r[1]-r[3]/2)*W, (r[2]-r[4]/2)*H, (r[1]+r[3]/2)*W, (r[2]+r[4]/2)*H]
            for r in gt.get(n, []) if r[0] == cls]

def dashed(d, box, color, width, dash=18, gap=12):
    x0, y0, x1, y1 = box
    for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        L = max(abs(bx-ax), abs(by-ay))
        if L == 0:
            continue
        ux, uy = (bx-ax)/L, (by-ay)/L
        t = 0.0
        while t < L:
            e = min(t+dash, L)
            d.line([ax+ux*t, ay+uy*t, ax+ux*e, ay+uy*e], fill=color, width=width)
            t = e + gap

def chip(d, xy, text, bg, fg, f, pad=(8, 4), anchor='lt'):
    bb = d.textbbox((0, 0), text, font=f)
    w, h = bb[2]-bb[0]+pad[0]*2, bb[3]-bb[1]+pad[1]*2
    x, y = xy
    if anchor == 'rt':
        x -= w
    if anchor == 'lb':
        y -= h
    d.rectangle([x, y, x+w, y+h], fill=bg)
    d.text((x+pad[0]-bb[0], y+pad[1]-bb[1]), text, font=f, fill=fg)
    return w, h

def draw_frame(n, scale=1.0, lw=3, fs=20, inset=True):
    """Render one frame with OOF predictions and GT mite boxes. Returns PIL image."""
    im = load(n).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    f = F(fs, bold=True)
    preds = sorted([r for r in R.get(n, []) if r[1] >= CONF], key=lambda r: r[1])
    for r in preds:
        x0, y0, x1, y1 = r[2:6]
        d.rectangle([x0-1, y0-1, x1+1, y1+1], outline=WHITE, width=lw+2)   # surface ring
        d.rectangle([x0, y0, x1, y1], outline=BLUE, width=lw)
        lab = f"Non-mite {r[1]:.2f}"
        ly = y0 - 2
        cw, ch = chip(d, (x0, max(0, ly)), lab, BLUE, WHITE, f, anchor='lb' if ly > 28 else 'lt')
    gtm = gtbox(n, 1, W, H)
    # when the GT bee is matched only below the display threshold, show that box explicitly
    note = None
    if gtm:
        def _iou(a, b):
            x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
            it = max(0, x2-x1)*max(0, y2-y1)
            return it/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it+1e-9)
        allp = R.get(n, [])
        bm = max(allp, key=lambda q: _iou(gtm[0], q[2:6]), default=None)
        if bm is not None:
            bi = _iou(gtm[0], bm[2:6])
            note = f"best-matching prediction: non-mite, conf {bm[1]:.3f}, IoU {bi:.2f}"
            if bm[1] < CONF and bi >= 0.3:
                x0, y0, x1, y1 = bm[2:6]
                dashed(d, (x0, y0, x1, y1), BLUE, max(2, lw-1), dash=14, gap=12)
                chip(d, (x0, max(0, y0-2)), f"Non-mite {bm[1]:.3f} (below display threshold)",
                     BLUE, WHITE, F(max(14, fs-6), True), anchor='lb' if y0 > 28 else 'lt')
    for g in gtm:
        d.rectangle([g[0]-2, g[1]-2, g[2]+2, g[3]+2], outline=WHITE, width=lw+3)
        dashed(d, g, RED, lw+1)
        chip(d, (g[0], min(H-30, g[3]+4)), "GT: bee with visible mite", RED, WHITE, f)
    mites = read_mite(n, W, H)
    for mb in mites:                      # tight mite-level box, solid
        d.rectangle([mb[0]-2, mb[1]-2, mb[2]+2, mb[3]+2], outline=WHITE, width=lw+3)
        d.rectangle(mb, outline=RED, width=lw)
    if inset and mites:
        mb = mites[0]
        pad = 0.9*max(mb[2]-mb[0], mb[3]-mb[1])
        cx0, cy0 = max(0, int(mb[0]-pad)), max(0, int(mb[1]-pad))
        cx1, cy1 = min(W, int(mb[2]+pad)), min(H, int(mb[3]+pad))
        crop = load(n).convert("RGB").crop((cx0, cy0, cx1, cy1))
        target = W*0.30
        k = target/crop.width
        crop = crop.resize((int(target), max(1, int(crop.height*k))), Image.LANCZOS)
        cd = ImageDraw.Draw(crop)
        cd.rectangle([(mb[0]-cx0)*k, (mb[1]-cy0)*k, (mb[2]-cx0)*k, (mb[3]-cy0)*k],
                     outline=RED, width=max(3, lw))
        occupied = [gtm[0] if gtm else mb] + [r[2:6] for r in preds]
        M = 14
        pos, bestov = None, None
        for px, py in ((M, M), (W-crop.width-M, M),
                       (M, H-crop.height-M), (W-crop.width-M, H-crop.height-M)):
            box = (px, py, px+crop.width, py+crop.height)
            ov = sum(max(0, min(box[2], o[2])-max(box[0], o[0])) *
                     max(0, min(box[3], o[3])-max(box[1], o[1])) for o in occupied)
            if bestov is None or ov < bestov:
                pos, bestov = (px, py), ov
        px, py = pos
        mcx, mcy = (mb[0]+mb[2])/2, (mb[1]+mb[3])/2
        acx = px if mcx > px+crop.width/2 else px+crop.width
        acy = py if mcy > py+crop.height/2 else py+crop.height
        d.line([acx, acy, mcx, mcy], fill=WHITE, width=lw+3)
        d.line([acx, acy, mcx, mcy], fill=RED, width=max(2, lw-1))
        im.paste(crop, (int(px), int(py)))
        d.rectangle([px-4, py-4, px+crop.width+4, py+crop.height+4], outline=WHITE, width=8)
        d.rectangle([px-3, py-3, px+crop.width+3, py+crop.height+3], outline=RED, width=5)
        chip(d, (px, py+crop.height+7),
             f"mite, magnified \u00d7{k:.0f}", RED, WHITE, F(max(16, fs-6), True))
    if scale != 1.0:
        im = im.resize((int(W*scale), int(H*scale)), Image.LANCZOS)
    return im, note

# ---------------- Figure 7 ----------------
n = FIG7
img, _ = draw_frame(n, lw=4, fs=26, inset=False)
W, H = img.size
FH = 86
canvas = Image.new("RGB", (W, H+FH), WHITE)
canvas.paste(img, (0, 0))
d = ImageDraw.Draw(canvas)
fb, fr = F(26, True), F(24)
y = H + 16
w, _ = chip(d, (14, y), "No ground-truth mite in this frame; none predicted", INK2, WHITE, fb)
d.text((14+w+16, y+4), f"{MODEL}, out-of-fold (fold {folds[n]})  |  conf \u2265 {CONF:.2f}  |  NMS IoU 0.45  |  "
       f"{stamp(n)}", font=fr, fill=INK2)
p7 = os.path.join(OUT, "Figure7_detection.png")
canvas.save(p7)
print("Figure 7:", p7, canvas.size, "| frame:", n, "fold", folds[n])

# ---------------- Figure 8 ----------------
PW, PH = 960, 540
HB = 44
names = FIG8_FN + FIG8_TN
cells = []
for i, nm in enumerate(names):
    body, note = draw_frame(nm, scale=PW/1920, lw=6, fs=32, inset=(i < 3))
    cell = Image.new("RGB", (PW, HB+PH), WHITE)
    cell.paste(body, (0, HB))
    cd = ImageDraw.Draw(cell)
    if note:
        chip(cd, (8, HB+PH-8), note, INK, WHITE, F(17, True), anchor='lb')
    cd.rectangle([0, 0, PW, HB], fill=INK)
    fh = F(19, True)
    cd.text((10, 12), f"({chr(97+i)})  {stamp(nm)}", font=fh, fill=WHITE)
    isfn = i < 3
    chip(cd, (PW-10, 8), "FN  mite present, not detected" if isfn else "TN  no mite present or predicted",
         RED if isfn else INK2, WHITE, F(17, True), anchor='rt')
    cells.append(cell)

GAP = 10
LEG = 136
GW = PW*3 + GAP*2
GH = (HB+PH)*2 + GAP
fig = Image.new("RGB", (GW, GH+LEG), WHITE)
for i, c in enumerate(cells):
    fig.paste(c, ((i % 3)*(PW+GAP), (i//3)*(HB+PH+GAP)))
d = ImageDraw.Draw(fig)
y = GH + 20
fl, fs_ = F(21, True), F(20)
x = 14
d.rectangle([x, y+2, x+34, y+22], outline=BLUE, width=3)
d.text((x+46, y), "Predicted: bee without visible mite", font=fs_, fill=INK)
x += 430
dashed(d, (x, y+2, x+34, y+22), RED, 3, dash=9, gap=6)
d.text((x+46, y), "Ground truth: bee with visible mite", font=fs_, fill=INK)
x += 430
d.rectangle([x, y+2, x+34, y+22], outline=RED, width=3)
d.text((x+46, y), "Mite (annotated for this figure only)", font=fs_, fill=INK)
x += 450
chip(d, (x, y), "FN", RED, WHITE, fl)
d.text((x+52, y), "false negative", font=fs_, fill=INK)
x += 230
chip(d, (x, y), "TN", INK2, WHITE, fl)
d.text((x+52, y), "true negative", font=fs_, fill=INK)
d.text((14, y+40), f"{MODEL} (highest non-mite AP@0.5 of the eight architectures), out-of-fold: each frame "
                   f"was scored by the fold model that excluded it  |  conf \u2265 {CONF:.2f}  |  NMS IoU 0.45  |  "
                   f"in-hive frames, 1920 \u00d7 1080", font=fs_, fill=INK2)
d.text((14, y+68), "In (a)\u2013(c) a predicted box matches the mite-bearing bee geometrically (IoU 0.88, 0.88 and "
                   "0.96) but is assigned to the non-mite class; the match clears the 0.25 display threshold "
                   "only in (a) (conf 0.450), while in (b) and (c) the bee registers at only 0.009.",
       font=fs_, fill=INK2)
d.text((14, y+94), "No mite-class box was emitted anywhere in the three frames, even at the 0.001 confidence "
                   "floor. The tight mite boxes (69\u201375 \u00d7 51\u201355 px) were annotated at mite level "
                   "for this figure only; training used bee-level boxes (Section 3.4).",
       font=fs_, fill=INK2)
p8 = os.path.join(OUT, "Figure8_panels.png")
fig.save(p8)
print("Figure 8:", p8, fig.size)
for i, nm in enumerate(names):
    mc = max([r[1] for r in R.get(nm, []) if r[0] == 1], default=0.0)
    print(f"  ({chr(97+i)}) {nm:34s} fold {folds[nm]}  max mite-class conf = {mc:.4f}")
