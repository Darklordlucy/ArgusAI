# vita_guard/scripts/augment_dataset.py
import albumentations as A
import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

IMG_DIR = "Argus_AI/data/train/images"
LBL_DIR = "Argus_AI/data/train/labels"

# ── Augmentation pipelines — fixed for albumentations v2.x ───────────────────
AUGMENTATIONS = [
    ("rain", A.Compose([
        A.RandomRain(
            slant_range=(-10, 10),
            drop_length=20,
            drop_width=1,
            drop_color=(200, 200, 200),
            blur_value=3,
            brightness_coefficient=0.85,
            p=1.0
        ),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3
    ))),

    ("fog", A.Compose([
        A.RandomFog(
            fog_coef_range=(0.2, 0.5),   # ← v2.x uses fog_coef_range tuple
            alpha_coef=0.1,
            p=1.0
        ),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3
    ))),

    ("night", A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=(-0.6, -0.4),   # ← v2.x uses brightness_limit
            contrast_limit=(-0.1, 0.2),      # ← v2.x uses contrast_limit
            p=1.0
        ),
        A.GaussNoise(
            std_range=(0.05, 0.15),          # ← v2.x uses std_range instead of var_limit
            p=0.7
        ),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3
    ))),

    ("blur", A.Compose([
        A.MotionBlur(
            blur_limit=(5, 9),
            p=1.0
        ),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3
    ))),
]


def read_yolo_label(lbl_path):
    bboxes, classes = [], []
    if not os.path.exists(lbl_path):
        return bboxes, classes
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    cid = int(parts[0])
                    box = [float(p) for p in parts[1:]]
                    box = [max(0.001, min(0.999, v)) for v in box]
                    if box[2] > 0.005 and box[3] > 0.005:
                        classes.append(cid)
                        bboxes.append(box)
                except ValueError:
                    continue
    return bboxes, classes


# Only process original images — skip already augmented ones
aug_suffixes    = ['_rain', '_fog', '_night', '_blur']
all_images      = list(Path(IMG_DIR).glob("*.jpg")) + \
                  list(Path(IMG_DIR).glob("*.png"))
original_images = [
    p for p in all_images
    if not any(s in p.stem for s in aug_suffixes)
]

print(f"Albumentations version: {A.__version__}")
print(f"Original images found:  {len(original_images)}")
print(f"Will generate:          {len(original_images) * 4} augmented images")
print(f"Total after:            {len(original_images) * 5}\n")

success = 0
skipped = 0
failed  = 0

for img_path in tqdm(original_images, desc="Augmenting", unit="img"):
    lbl_path            = os.path.join(LBL_DIR, img_path.stem + ".txt")
    bboxes, classes     = read_yolo_label(lbl_path)

    if not bboxes:
        skipped += 1
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        skipped += 1
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for aug_name, transform in AUGMENTATIONS:
        try:
            result      = transform(
                image        = img_rgb,
                bboxes       = bboxes,
                class_labels = classes
            )
            aug_img     = cv2.cvtColor(result['image'], cv2.COLOR_RGB2BGR)
            aug_boxes   = result['bboxes']
            aug_classes = result['class_labels']

            if not aug_boxes:
                continue

            # Save augmented image
            out_img = str(img_path).replace(
                img_path.name,
                img_path.stem + f"_{aug_name}" + img_path.suffix
            )
            cv2.imwrite(out_img, aug_img)

            # Save augmented label
            out_lbl = os.path.join(
                LBL_DIR,
                img_path.stem + f"_{aug_name}.txt"
            )
            with open(out_lbl, 'w') as f:
                for cls, box in zip(aug_classes, aug_boxes):
                    f.write(
                        f"{cls} {box[0]:.6f} {box[1]:.6f} "
                        f"{box[2]:.6f} {box[3]:.6f}\n"
                    )

            success += 1

        except Exception as e:
            failed += 1
            # Uncomment to debug a specific error:
            # print(f"\nError on {img_path.name} / {aug_name}: {e}")
            continue

print(f"\nDone!")
print(f"  Created:  {success} augmented images")
print(f"  Skipped:  {skipped} (no labels or corrupt image)")
print(f"  Failed:   {failed}  (transform errors)")
print(f"\nTotal training images now: {len(list(Path(IMG_DIR).glob('*.*')))}")