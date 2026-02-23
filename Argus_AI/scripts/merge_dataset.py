# vita_guard/scripts/merge_datasets.py
import os
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm

OUT_DIR = Path("Argus_AI/data")
BASE    = Path("Argus_AI/data/raw_downloads")

# ── Class remapping rules ─────────────────────────────────────────────────────
# YOUR classes: 0=pothole, 1=pedestrian, 2=obstacle
#
# For each dataset, define how to map THEIR class IDs to YOUR class IDs.
# "all_to_0" = every class in that dataset becomes pothole
# "all_to_1" = every class becomes pedestrian
# "all_to_2" = every class becomes obstacle
# OR specify a dict: {their_class_id: your_class_id, ...}
#
# The ADP traffic dataset (traffic_combined) has these classes in order:
#   0=Car, 1=Cyclist, 2=DontCare, 3=Misc, 4=Pedestrian, 5=Person_sitting,
#   6=Tram, 7=Truck, 8=Van
# Check your actual class names using check_classes.py and update below

DATASET_CONFIGS = [
    {
        "folder":       "pothole_main",
        "class_remap":  "all_to_0",   # All damage classes → pothole
        "splits":       {"train": "train", "val": "valid"},
    },
    {
        "folder":       "pothole_extra",
        "class_remap":  "all_to_0",
        "splits":       {"train": "train", "val": "valid"},
    },
    {
        "folder":       "pedestrian_main",
        "class_remap":  "all_to_1",   # All classes = person = pedestrian
        "splits":       {"train": "train", "val": "valid"},
    },
    {
        "folder":       "traffic_combined",
        # ADP dataset classes (verify with check_classes.py first):
        # 0=Car→2, 1=Cyclist→1, 2=DontCare→SKIP, 3=Misc→SKIP,
        # 4=Pedestrian→1, 5=Person_sitting→1, 6=Tram→2, 7=Truck→2, 8=Van→2
        "class_remap":  {
            0: 2,   # Car      → obstacle
            1: 1,   # Cyclist  → pedestrian
            2: -1,  # DontCare → SKIP (use -1 to skip)
            3: -1,  # Misc     → SKIP
            4: 1,   # Pedestrian → pedestrian
            5: 1,   # Person_sitting → pedestrian
            6: 2,   # Tram     → obstacle
            7: 2,   # Truck    → obstacle
            8: 2,   # Van      → obstacle
        },
        "splits":       {"train": "train", "val": "valid"},
    },
    {
        "folder":       "vehicles",
        "class_remap":  "all_to_2",   # All vehicle types → obstacle
        "splits":       {"train": "train", "val": "valid"},
    },
]


def get_remapped_lines(label_path, class_remap):
    """Read YOLO label file, remap class IDs, return valid lines"""
    new_lines = []
    try:
        with open(label_path) as f:
            raw_lines = f.readlines()
    except:
        return []

    for line in raw_lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        try:
            orig_class = int(parts[0])
            cx  = float(parts[1])
            cy  = float(parts[2])
            bw  = float(parts[3])
            bh  = float(parts[4])
        except ValueError:
            continue

        # Validate bounding box
        if not (0 <= cx <= 1 and 0 <= cy <= 1):
            continue
        if not (0.005 < bw <= 1 and 0.005 < bh <= 1):
            continue

        # Get new class ID
        if class_remap == "all_to_0":
            new_class = 0
        elif class_remap == "all_to_1":
            new_class = 1
        elif class_remap == "all_to_2":
            new_class = 2
        elif isinstance(class_remap, dict):
            new_class = class_remap.get(orig_class, -1)
        else:
            continue

        if new_class == -1:
            continue   # Skip this annotation

        new_lines.append(
            f"{new_class} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        )

    return new_lines


# ── Run the merge ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Argus_AI Dataset Merge")
print("=" * 60)

grand_total = 0

for config in DATASET_CONFIGS:
    folder_name = config["folder"]
    class_remap = config["class_remap"]
    splits      = config["splits"]

    dataset_path = BASE / folder_name
    if not dataset_path.exists():
        print(f"\n⚠ SKIP: {folder_name} — folder not found")
        continue

    print(f"\nProcessing: {folder_name}")
    dataset_total = 0

    for your_split, their_split in splits.items():
        img_src = dataset_path / their_split / "images"
        lbl_src = dataset_path / their_split / "labels"

        # Also check if split folder is directly inside (some datasets skip subfolder)
        if not img_src.exists():
            img_src = dataset_path / "images"
            lbl_src = dataset_path / "labels"
        if not img_src.exists():
            print(f"  SKIP split '{their_split}' — images folder not found")
            continue

        img_dst = OUT_DIR / your_split / "images"
        lbl_dst = OUT_DIR / your_split / "labels"

        # Get all images
        images = list(img_src.glob("*.jpg")) + \
                 list(img_src.glob("*.png")) + \
                 list(img_src.glob("*.jpeg"))

        if not images:
            print(f"  SKIP split '{their_split}' — no images found")
            continue

        count = 0
        for img_path in tqdm(images, desc=f"  {their_split}→{your_split}",
                             unit="img", leave=False):
            # Find label file
            lbl_path = lbl_src / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            # Remap class IDs
            new_lines = get_remapped_lines(str(lbl_path), class_remap)
            if not new_lines:
                continue

            # Create unique filename — prefix with dataset name to avoid collisions
            unique_stem = f"{folder_name}_{img_path.stem}"
            unique_img  = unique_stem + img_path.suffix

            # Copy image
            shutil.copy2(str(img_path), str(img_dst / unique_img))

            # Write remapped label
            with open(str(lbl_dst / (unique_stem + ".txt")), 'w') as f:
                f.write('\n'.join(new_lines) + '\n')

            count += 1

        print(f"  ✓ {their_split} → {your_split}: {count} images")
        dataset_total += count

    print(f"  Dataset total: {dataset_total}")
    grand_total += dataset_total


# ── Final count ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MERGE COMPLETE")
print("=" * 60)
print(f"Grand total images merged: {grand_total}")
print()
for split in ["train", "val"]:
    imgs = len(list((OUT_DIR / split / "images").glob("*.*")))
    lbls = len(list((OUT_DIR / split / "labels").glob("*.txt")))
    match = "✓ OK" if imgs == lbls else f"✗ MISMATCH ({abs(imgs-lbls)} difference)"
    print(f"  {split:6s}: {imgs:5d} images | {lbls:5d} labels | {match}")
print("=" * 60)

