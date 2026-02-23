# Argus_AI/scripts/more_potholes.py
from roboflow import Roboflow
from pathlib import Path
import shutil, os
from tqdm import tqdm

API_KEY = "Vs168lA80zUFL6DL6F5V" 

BASE_RAW      = Path("Argus_AI/data/raw_downloads")
OUT_TRAIN_IMG = Path("Argus_AI/data/train/images")
OUT_TRAIN_LBL = Path("Argus_AI/data/train/labels")
BASE_RAW.mkdir(parents=True, exist_ok=True)

# ── Verified dataset URLs from Roboflow Universe (searched and confirmed) ─────
# workspace/project come directly from universe.roboflow.com/workspace/project
DATASETS = [
    {
        "name":      "GeraPotHole (608 images)",
        "workspace": "gerapothole",
        "project":   "pothole-detection-yolov8",
        "version":   1,
        "folder":    "pothole_gera",
    },
    {
        "name":      "Matt Pothole (660 images)",
        "workspace": "matt-s1uw6",
        "project":   "pothole-detection-yolov8-ehkp9",
        "version":   1,
        "folder":    "pothole_matt",
    },
    {
        "name":      "Kartik Pothole (608 images)",
        "workspace": "kartik-zvust",
        "project":   "pothole-detection-yolo-v8",
        "version":   1,
        "folder":    "pothole_kartik",
    },
    {
        "name":      "KUET Pothole (327 images)",
        "workspace": "kuet-hetlo",
        "project":   "pothole-yolov8",
        "version":   1,
        "folder":    "pothole_kuet",
    },
    {
        "name":      "YOLOv8 UAV Potholes",
        "workspace": "yolov8-uav",
        "project":   "potholes-detect-uytky",
        "version":   1,
        "folder":    "pothole_uav",
    },
    {
        "name":      "YoloV8 Pothole (262 images)",
        "workspace": "yolov8-ev9lt",
        "project":   "pothole-detection-wh3mk",
        "version":   1,
        "folder":    "pothole_ev9lt",
    },
]
# ─────────────────────────────────────────────────────────────────────────────

def merge_into_train(folder_path, folder_name):
    """Read all images in a downloaded dataset, remap all classes → 0, copy to train."""
    folder = Path(folder_path)
    merged = 0

    # Search for images in train/ valid/ or root images/
    search_dirs = [
        folder / "train" / "images",
        folder / "valid" / "images",
        folder / "images",
    ]

    for img_dir in search_dirs:
        if not img_dir.exists():
            continue

        lbl_dir = img_dir.parent / "labels"
        if not lbl_dir.exists():
            continue

        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        for img_path in tqdm(images, desc=f"  Merging {img_dir.parent.name}",
                             unit="img", leave=False):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            # Read label and remap ALL classes → 0 (pothole)
            new_lines = []
            with open(lbl_path, encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    try:
                        cx  = float(parts[1])
                        cy  = float(parts[2])
                        bw  = float(parts[3])
                        bh  = float(parts[4])
                        # Only keep boxes with valid size
                        if 0 < bw <= 1 and 0 < bh <= 1 and 0 <= cx <= 1 and 0 <= cy <= 1:
                            new_lines.append(
                                f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                            )
                    except ValueError:
                        continue

            if not new_lines:
                continue

            # Unique filename — prefix with folder name to avoid collisions
            unique_stem = f"{folder_name}_{img_path.stem}"
            unique_img  = unique_stem + img_path.suffix

            shutil.copy2(str(img_path), str(OUT_TRAIN_IMG / unique_img))

            with open(str(OUT_TRAIN_LBL / (unique_stem + ".txt")), 'w') as f:
                f.write('\n'.join(new_lines) + '\n')

            merged += 1

    return merged


print("=" * 60)
print("Downloading Extra Pothole Datasets")
print("=" * 60)

rf = Roboflow(api_key=API_KEY)

total_added = 0
success_count = 0
failed_list   = []

for ds in DATASETS:
    print(f"\n{'─'*60}")
    print(f"Dataset: {ds['name']}")
    print(f"  workspace: {ds['workspace']}")
    print(f"  project:   {ds['project']}")

    dest = str(BASE_RAW / ds['folder'])

    # Try version 1, then 2, then 3
    downloaded = False
    for ver in [1, 2, 3]:
        try:
            print(f"  Trying version {ver}...")
            proj    = rf.workspace(ds['workspace']).project(ds['project'])
            dataset = proj.version(ver).download(
                "yolov8",
                location=dest,
                overwrite=True
            )
            print(f"  ✓ Download successful (version {ver})")
            downloaded = True
            break
        except Exception as e:
            err = str(e)
            if "zip" in err.lower():
                # Download succeeded but zip is empty/corrupt — still try to merge
                print(f"  ⚠ Version {ver}: zip issue — attempting merge anyway")
                downloaded = True
                break
            elif "404" in err or "does not exist" in err:
                print(f"  Version {ver}: not found — trying next version")
                continue
            else:
                print(f"  Version {ver} error: {err[:80]}")
                continue

    if not downloaded:
        print(f"  ✗ All versions failed — adding to manual list")
        failed_list.append(ds)
        continue

    # Merge into training set
    added = merge_into_train(dest, ds['folder'])
    print(f"  ✓ Merged {added} images into training set")
    total_added   += added
    success_count += 1


# ── Final count ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"Datasets downloaded: {success_count}/{len(DATASETS)}")
print(f"New images added:    {total_added}")

# Count class 0 in full training set
count_0 = 0
for lbl in Path("Argus_AI/data/train/labels").glob("*.txt"):
    for line in open(lbl, encoding='utf-8', errors='ignore'):
        line = line.strip()
        if line:
            try:
                if int(float(line.split()[0])) == 0:
                    count_0 += 1
            except:
                continue

print(f"Total pothole (class 0) annotations: {count_0}")

if failed_list:
    print(f"\n{'─'*60}")
    print("MANUAL DOWNLOAD NEEDED for these datasets:")
    print("Go to each URL, click Download → YOLOv8 → Show Code\n")
    for ds in failed_list:
        url = f"universe.roboflow.com/{ds['workspace']}/{ds['project']}"
        print(f"  {ds['name']}")
        print(f"  URL: {url}\n")

print(f"{'='*60}")