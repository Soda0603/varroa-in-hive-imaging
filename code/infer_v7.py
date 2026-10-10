import os, sys, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8', errors='replace')
REPO = os.environ.get("YOLOV7_REPO", "yolov7")
sys.path.insert(0, REPO)
_cwd = os.getcwd()
os.chdir(REPO)
import torch, numpy as np, cv2
_o = torch.load
torch.load = lambda *a, **k: _o(*a, **{**k, 'weights_only': False})
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

BASE = os.environ.get("VARROA_DATA", "data")
MW = os.path.join(os.environ.get("VARROA_WEIGHTS", "weights"), "yolov7_5fold_weights")
ALLD = os.path.join(BASE, "mydataset", "all")
OUT = os.path.join(os.environ.get("VARROA_OUT", "out"), "oof2823")
os.makedirs(OUT, exist_ok=True)

def _readlines(p):
    out = []
    for b in open(p, 'rb').read().splitlines():
        b = b.strip()
        if not b:
            continue
        for enc in ('utf-8', 'cp950', 'big5'):
            try:
                out.append(b.decode(enc)); break
            except UnicodeDecodeError:
                continue
    return out

folds = {}
for i in range(1, 6):
    for l in _readlines(os.path.join(BASE, "5-Fold_Stratified", f"fold-{i}", "val.txt")):
        folds[os.path.splitext(os.path.basename(l))[0]] = i
print("OOF images:", len(folds), flush=True)

def imgpath(n):
    for ext in ('.jpg', '.jpeg', '.png', '.JPG'):
        p = os.path.join(ALLD, n + ext)
        if os.path.exists(p):
            return p

dev = torch.device('cuda:0')
rec = {}
t0 = time.time()
for i in range(1, 6):
    wp = os.path.join(MW, f"fold-{i}", "best.pt")
    model = attempt_load(wp, map_location=dev); model.eval()
    names = [n for n, f in folds.items() if f == i]
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
                d = det.cpu().numpy()
                for *xy, cf, cl in d:
                    rows.append([int(cl), round(float(cf), 5),
                                 round(float(xy[0]), 1), round(float(xy[1]), 1),
                                 round(float(xy[2]), 1), round(float(xy[3]), 1)])
            rec[n] = rows
    print(f"  YOLOv7 fold-{i}: {len(names)} images", flush=True)
    del model; torch.cuda.empty_cache()
json.dump(rec, open(os.path.join(OUT, "YOLOv7.json"), "w"))
print(f"YOLOv7: {len(rec)} images, {time.time()-t0:.0f}s", flush=True)
print("V7 DONE", flush=True)
