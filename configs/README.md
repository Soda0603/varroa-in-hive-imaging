# Training configurations

These files are kept **exactly as they were used**, absolute paths and all, so that the
training runs behind the published tables stay auditable. They are a record, not a template.

To rerun anything, edit these fields first:

**`fold-N-ultralytics.yaml`** (YOLOv8n, YOLOv9c, YOLOv10n, YOLO11n, YOLO12n, YOLOv13n, YOLO26n)

| Field | Change to |
|---|---|
| `path` | the root of the unzipped Zenodo dataset, the folder containing `mydataset/all` |
| `train` | your copy of `../splits/fold-N-train.txt` |
| `val` | your copy of `../splits/fold-N-val.txt` |

The split files in `../splits/` use paths relative to `path`
(`./mydataset/all/<image>.jpg`), so they work unchanged once `path` is right.

**`fold-N-yolov7.yaml`**, **`yolov7-opt.yaml`**, **`yolov7-hyp.yaml`**

`cfg` and `hyp` point into a clone of [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7).
Set them to your own clone, or set `YOLOV7_REPO` and use the scripts in `../code/`.

**`*-args.yaml`** are the resolved ultralytics argument dumps for each architecture. They
record the hyperparameters that were actually in force, including the ones left at their
defaults.

Classes are the same throughout: `0 = Bee without visible mite`, `1 = Bee with visible mite`.
The box marks the bee, not the mite.
