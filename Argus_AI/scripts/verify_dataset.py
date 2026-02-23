# vita_guard/scripts/verify_dataset.py
import os
import cv2
from pathlib import Path

def verify_split(split_name):
    base     = Path("Argus_AI/data")
    img_dir  = base / split_name / "images"
    lbl_dir  = base / split_name / "labels"

    images   = list(img_dir.glob("*.jpg")) + \
               list(img_dir.glob("*.png")) + \
               list(img_dir.glob("*.jpeg"))

    print(f"\n{'='*55}")
    print(f"  Verifying: {split_name}  ({len(images)} images)")
    print(f"{'='*55}")

    errors         = []
    class_counts   = {0: 0, 1: 0, 2: 0}
    total_boxes    = 0

    for img_path in images:
        # 1. Check image is not corrupt
        img = cv2.imread(str(img_path))
        if img is None:
            errors.append(f"CORRUPT: {img_path.name}")
            continue

        # 2. Check image is not too small
        h, w = img.shape[:2]
        if w < 32 or h < 32:
            errors.append(f"TOO SMALL ({w}x{h}): {img_path.name}")
            continue

        # 3. Check label file exists
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            errors.append(f"NO LABEL: {img_path.stem}.txt")
            continue

        # 4. Validate every line in the label file
        with open(lbl_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            errors.append(f"EMPTY LABEL: {lbl_path.name}")
            continue

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"BAD FORMAT in {lbl_path.name}: '{line}'")
                continue
            try:
                cid  = int(parts[0])
                cx   = float(parts[1])
                cy   = float(parts[2])
                bw   = float(parts[3])
                bh   = float(parts[4])

                if cid not in [0, 1, 2]:
                    errors.append(f"INVALID CLASS {cid} in {lbl_path.name}")
                    continue
                if not (0 <= cx <= 1 and 0 <= cy <= 1):
                    errors.append(f"CENTER OUT OF RANGE in {lbl_path.name}")
                    continue
                if not (0 < bw <= 1 and 0 < bh <= 1):
                    errors.append(f"SIZE OUT OF RANGE in {lbl_path.name}")
                    continue

                class_counts[cid] += 1
                total_boxes += 1

            except ValueError:
                errors.append(f"NON-NUMERIC in {lbl_path.name}: '{line}'")

    # Print class distribution
    print(f"\n  Class Distribution ({total_boxes} total annotations):")
    names = {0: "pothole", 1: "pedestrian", 2: "obstacle"}
    for cid, count in class_counts.items():
        pct  = (count / total_boxes * 100) if total_boxes > 0 else 0
        bar  = "█" * int(pct / 2)
        flag = "  ⚠ LOW" if count < 100 else ""
        print(f"    Class {cid} ({names[cid]:12s}): {count:5d}  {pct:5.1f}%  {bar}{flag}")

    # Print errors
    if errors:
        print(f"\n  ✗ ERRORS FOUND: {len(errors)}")
        for e in errors[:15]:
            print(f"      {e}")
        if len(errors) > 15:
            print(f"      ... and {len(errors) - 15} more")
    else:
        print(f"\n  ✓ All {len(images)} images clean — no errors")

    return errors, class_counts


print("\nVITA-Guard Dataset Verification")

train_errors, train_counts = verify_split("train")
val_errors,   val_counts   = verify_split("val")

total_errors = len(train_errors) + len(val_errors)

print(f"\n{'='*55}")
if total_errors == 0:
    print("  ✓ DATASET IS CLEAN — Ready for augmentation")
else:
    print(f"  ✗ {total_errors} errors found — fix before continuing")
print(f"{'='*55}\n")