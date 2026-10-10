# In-hive *Varroa destructor* imaging - splits, labels, verification and evaluation code

Code and small data for the study *A sliding-rail in-hive imaging platform for
Varroa destructor surveillance: an open dataset and the failure of eight YOLO
architectures at field prevalence*.

**The images and the trained weights are not in this repository.** They are too large for
GitHub and live on Zenodo:

> **https://doi.org/10.5281/zenodo.23099411**
> `dataset_2823_images_labels.zip` (2.0 GB) and `weights.zip` (734 MB)

Cite the Zenodo concept DOI above; it always resolves to the latest version.

## What is here

| Path | Contents |
|---|---|
| `splits/` | five-fold stratified split definitions, seed 42, UTF-8 (`fold-N-train.txt`, `fold-N-val.txt`) |
| `mite_level_labels/` | 69 tight mite boxes, single class, a separate layer from the original labels |
| `verification/` | per-annotation re-inspection record for all 87 class-1 boxes |
| `code/` | training, inference, evaluation and figure code - see `code/README.txt` |
| `configs/` | training YAML for the eight architectures |

## The dataset in one table

| Source | Images | Class-0 boxes | Class-1 boxes | Mite-bearing images |
|---|---|---|---|---|
| In-hive sliding-rail camera (1920 x 1080) | 2,795 | 4,581 | 23 | 23 |
| Supplementary handheld close-range photographs | 28 | 111 | 64 | 26 |
| **Total** | **2,823** | **4,692** | **87** | **49** |

Classes: `0 = Bee without visible mite`, `1 = Bee with visible mite`. The box marks the
**bee**, not the mite, and "without visible mite" does not mean the bee is uninfested.

## Two things to read before you train on this

**The mite class is very weakly supported.** Mite-level re-inspection confirmed 56 of the 87
class-1 annotations, and only **3 of the 23** in-hive ones. 36 further mites in the
supplementary images were never annotated. Do not treat the in-hive class-1 labels as reliable
positives. In the accompanying study, none of eight architectures (2.5-37.2 M parameters)
detected any of the 23 in-hive mite-bearing frames, while bee detection was unaffected
(AP@0.5 0.420-0.499).

**EXIF orientation.** Thirteen supplementary photographs carry orientation 6. The labels are
normalised against the **stored** frame, which is what LabelImg writes. Readers that honour
EXIF - `cv2.imread`, and the ultralytics and YOLOv7 dataloaders - rotate those images out of
register with their labels, affecting 48 boxes (31 of them class-1, i.e. 36% of all class-1
annotations). Decode with `IMREAD_IGNORE_ORIENTATION` as `code/common.py` does, or strip the
orientation tags first.

## Running the code

No path is hardcoded. Every script reads its locations from the environment, each with a
relative-path default:

| Variable | Default | What it points at |
|---|---|---|
| `VARROA_DATA` | `data` | the unzipped Zenodo dataset (`mydataset/all` inside) |
| `VARROA_OUT` | `out` | where predictions, metrics and figures are written |
| `VARROA_WEIGHTS` | `weights` | the unzipped `weights.zip` (8 architectures x 5 folds) |
| `VARROA_RUNS` | `runs` | training run directories, if you retrain |
| `YOLOV7_REPO` | `yolov7` | a clone of WongKinYiu/yolov7, needed only for the YOLOv7 scripts |
| `VARROA_MITE_IMAGES` | `mite_images` | the three verified in-hive mite frames, for Figures 7-8 |

```bash
export VARROA_DATA=/path/to/dataset
export VARROA_OUT=/path/to/output
export VARROA_WEIGHTS=/path/to/weights
python code/infer_ultra.py      # out-of-fold inference, all 8 architectures x 5 folds
python code/analyze.py          # AP, mAP and the per-class tables
```

PowerShell uses `$env:VARROA_DATA = "D:\dataset"` instead of `export`.

Architectures evaluated: YOLOv7, YOLOv8n, YOLOv9c, YOLOv10n, YOLO11n, YOLO12n, YOLOv13n,
YOLO26n, under one five-fold protocol with strictly out-of-fold evaluation.

## Licence

CC BY 4.0, matching the Zenodo record. See `LICENSE`.

## Citation

See `CITATION.cff`, or cite the Zenodo concept DOI `10.5281/zenodo.23099411` and the accompanying
manuscript once it is published.
