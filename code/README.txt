Evaluation and figure code
==========================

common.py holds the shared loaders. Every script has absolute paths at the top that
must be edited for your machine.

IMPORTANT - image orientation. Thirteen of the supplementary photographs carry EXIF
orientation 6. The YOLO labels in this dataset are normalised against the STORED pixel
frame, which is what LabelImg writes. Readers that apply EXIF orientation (cv2.imread,
and the ultralytics and YOLOv7 dataloaders) will rotate those images out of register
with their labels. common.imread therefore decodes with IMREAD_IGNORE_ORIENTATION and
common.img_size returns the stored size. Use the same convention, or strip the EXIF
orientation tags before training.
