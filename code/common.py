import os, glob, re
import numpy as np, cv2

BASE = os.environ.get("VARROA_DATA", "data")
ALLD = os.path.join(BASE, "mydataset", "all")
SCR = os.environ.get("VARROA_OUT", "out")
OOF = os.path.join(SCR, "oof2823")


def readlines(p):
    """Split files mix UTF-8 (ASCII) and cp950 (Chinese) lines; decode per line."""
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
        else:
            out.append(b.decode('utf-8', errors='replace'))
    return out


def fold_map(which="new"):
    sub = "5-Fold_Stratified" if which == "new" else "5-Fold訓練"
    pat = (lambda i: os.path.join(BASE, sub, f"fold-{i}", "val.txt")) if which == "new" \
        else (lambda i: os.path.join(BASE, sub, f"train-{i}", "val.txt"))
    m = {}
    for i in range(1, 6):
        for l in readlines(pat(i)):
            m[os.path.splitext(os.path.basename(l))[0]] = i
    return m


def ground_truth():
    gt = {}
    for f in glob.glob(os.path.join(ALLD, "*.txt")):
        n = os.path.basename(f)[:-4]
        if n == "classes":
            continue
        rows = []
        for l in open(f, encoding='utf8', errors='replace'):
            p = l.split()
            if len(p) >= 5:
                rows.append([int(float(p[0]))] + [float(x) for x in p[1:5]])
        gt[n] = rows
    return gt


def imgpath(n, d=ALLD):
    for ext in ('.jpg', '.jpeg', '.png', '.JPG'):
        p = os.path.join(d, n + ext)
        if os.path.exists(p):
            return p


def imread(fp):
    """Decode in the STORED pixel frame, ignoring EXIF orientation.

    13 of the supplementary photographs carry EXIF orientation 6. LabelImg does not
    apply EXIF, so every YOLO label in this dataset is normalised against the stored
    frame. cv2.imdecode applies EXIF by default, which rotates those 13 images out of
    register with their labels - hence IMREAD_IGNORE_ORIENTATION.
    """
    return cv2.imdecode(np.fromfile(fp, dtype=np.uint8),
                        cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)


_SIZE_CACHE = {}
def img_size(name_or_path):
    """(width, height) in the SAME frame the annotations use.

    13 of the supplementary photographs carry EXIF orientation 6: their stored
    pixels are 4032x3024 but LabelImg - and cv2.imdecode - present them rotated,
    as 3024x4032, and the YOLO labels are normalised against that rotated view.
    PIL's Image.size reports the stored frame and must not be used here.
    """
    from PIL import Image
    p = name_or_path if os.path.sep in str(name_or_path) else imgpath(name_or_path)
    if p in _SIZE_CACHE:
        return _SIZE_CACHE[p]
    with Image.open(p) as im:
        w, h = im.size          # stored frame; EXIF orientation is deliberately ignored
    _SIZE_CACHE[p] = (w, h)
    return w, h


IN_HIVE_RE = re.compile(r'^img-\d{8}_\d{6}--\d+$')

MODEL_DIRS = {
    "YOLOv7":   "yolov7_5fold_weights",
    "YOLOv8n":  "yolov8n_5fold_weights",
    "YOLOv9c":  "yolov9c_5fold_weights",
    "YOLOv10n": "yolov10n_5fold_weights",
    "YOLO11n":  "yolo11n_5fold_weights",
    "YOLO12n":  "yolo12n_5fold_weights",
    "YOLOv13n": "yolov13n_5fold_weights",
    "YOLO26n":  "yolo26n_5fold_weights",
}
ORDER = ["YOLOv7", "YOLOv8n", "YOLOv9c", "YOLOv10n", "YOLO11n", "YOLO12n", "YOLOv13n", "YOLO26n"]
MW = os.environ.get("VARROA_WEIGHTS", "weights")
