from roboflow import Roboflow
import os
import sys

API_KEY = "Vs168lA80zUFL6DL6F5V" 

BASE_DIR = "Argus_AI/data/raw_downloads"
os.makedirs(BASE_DIR, exist_ok=True)

def download_dataset(rf, workspace, project, location, max_versions=4):
    for version in range(1, max_versions + 1):
        try:
            print(f"  Trying version {version}...")
            proj    = rf.workspace(workspace).project(project)
            dataset = proj.version(version).download(
                "yolov8",
                location=location,
                overwrite=True
            )
            print(f"Downloaded version {version} → {location}")
            return dataset
        except Exception as e:
            err = str(e)
            if "404" in err or "does not exist" in err.lower():
                print(f"  Version {version} not found — trying next")
                continue
            else:
                print(f"  Version {version} error: {e}")
                continue
    print(f"  ✗ All versions failed for {workspace}/{project}")
    return None


print("=" * 60)
print("Argus_AI Dataset Downloader")
print("=" * 60)

# Connect to Roboflow
print("\nConnecting to Roboflow --------------")
try:
    rf = Roboflow(api_key=API_KEY)
    print("Connected !!!\n")
except Exception as e:
    print(f"Cannot connect : {e}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 1 — Pothole Detection (Intel Unnati)
# URL: universe.roboflow.com/intel-unnati-training-program/pothole-detection-bqu6s
# ~2,500 images | Indian road conditions | 8 damage classes → all = pothole
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 60)
print("DATASET 1: Pothole Detection (Intel Unnati — 2.5k images)")
print("─" * 60)
ds1 = download_dataset(
    rf,
    workspace = "intel-unnati-training-program",
    project   = "pothole-detection-bqu6s",
    location  = f"{BASE_DIR}/pothole_main"
)
if ds1:
    print(f"  Location: {ds1.location}\n")
else:
    print("  FALLBACK: Trying secondary pothole dataset...\n")
    ds1 = download_dataset(
        rf,
        workspace = "shantanu-maity",
        project   = "potholes-detection-qwkkc",
        location  = f"{BASE_DIR}/pothole_main"
    )


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 2 — Extra Pothole Dataset (Kartik)
# URL: universe.roboflow.com/kartik-zvust/pothole-detection-yolo-v8
# ~600 images | Different pothole angles and lighting
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 60)
print("DATASET 2: Extra Pothole Dataset (608 images)")
print("─" * 60)
ds2 = download_dataset(
    rf,
    workspace = "kartik-zvust",
    project   = "pothole-detection-yolo-v8",
    location  = f"{BASE_DIR}/pothole_extra"
)
if ds2:
    print(f"  Location: {ds2.location}\n")
else:
    print("  FALLBACK: Trying GeraPotHole dataset...\n")
    ds2 = download_dataset(
        rf,
        workspace = "gerapothole",
        project   = "pothole-detection-yolov8",
        location  = f"{BASE_DIR}/pothole_extra"
    )


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 3 — Pedestrian Detection (Yolov8 Training)
# URL: universe.roboflow.com/yolov8-training-l1ktn/pedestrians-ihyip
# ~385 images | Single class: Person → your class 1
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 60)
print("DATASET 3: Pedestrian Detection (~385 images)")
print("─" * 60)
ds3 = download_dataset(
    rf,
    workspace = "yolov8-training-l1ktn",
    project   = "pedestrians-ihyip",
    location  = f"{BASE_DIR}/pedestrian_main"
)
if ds3:
    print(f"  Location: {ds3.location}\n")
else:
    print("  FALLBACK: Trying pedestrian-w888x dataset...\n")
    ds3 = download_dataset(
        rf,
        workspace = "yolov8-hvfsr",
        project   = "pedestrian-w888x",
        location  = f"{BASE_DIR}/pedestrian_main"
    )


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 4 — Combined Road Dataset: Pedestrian + Vehicle + Cyclist
# URL: universe.roboflow.com/adp-l8hde/yolov8-6apfg
# ~7,500 images | Classes: Car, Van, Truck, Pedestrian, Cyclist
# Best dataset — covers both class 1 and class 2
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 60)
print("DATASET 4: Road Traffic Dataset — Pedestrian + Vehicles (7.5k images)")
print("─" * 60)
ds4 = download_dataset(
    rf,
    workspace = "adp-l8hde",
    project   = "yolov8-6apfg",
    location  = f"{BASE_DIR}/traffic_combined"
)
if ds4:
    print(f"  Location: {ds4.location}\n")
else:
    print("  FALLBACK: Trying public road tool dataset...\n")
    ds4 = download_dataset(
        rf,
        workspace = "public-road-tool",
        project   = "yolov8-b1edy",
        location  = f"{BASE_DIR}/traffic_combined"
    )


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 5 — Vehicle Detection (bicycle, bus, car, truck, two-wheeler, van)
# URL: universe.roboflow.com/object-detect-ydedz/vehicle-detection-2.0-wwhpg
# Covers two-wheelers specifically — important for Indian road context
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 60)
print("DATASET 5: Vehicle Detection 2.0 — includes two-wheelers")
print("─" * 60)
ds5 = download_dataset(
    rf,
    workspace = "object-detect-ydedz",
    project   = "vehicle-detection-2.0-wwhpg",
    location  = f"{BASE_DIR}/vehicles"
)
if ds5:
    print(f"  Location: {ds5.location}\n")
else:
    print("  FALLBACK: Trying ANPR vehicle dataset...\n")
    ds5 = download_dataset(
        rf,
        workspace = "anpr-yolov8",
        project   = "vehicle-detection-ovdwb",
        location  = f"{BASE_DIR}/vehicles"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DOWNLOAD SUMMARY")
print("=" * 60)

import pathlib
downloaded = []
failed     = []

checks = [
    ("pothole_main",     ds1),
    ("pothole_extra",    ds2),
    ("pedestrian_main",  ds3),
    ("traffic_combined", ds4),
    ("vehicles",         ds5),
]

for name, ds in checks:
    path = pathlib.Path(f"{BASE_DIR}/{name}")
    if path.exists():
        imgs = list(path.rglob("*.jpg")) + list(path.rglob("*.png"))
        print(f"  ✓ {name:<22} {len(imgs)} images found")
        downloaded.append(name)
    else:
        print(f"  ✗ {name:<22} FAILED — not downloaded")
        failed.append(name)

print(f"\n  Downloaded: {len(downloaded)}/5 datasets")
if failed:
    print(f"  Failed:     {failed}")
    print("\n  For failed datasets, manually download from Roboflow Universe:")
    print("  → Go to universe.roboflow.com")
    print("  → Search the dataset name")
    print("  → Click Download → YOLOv8 → Show Download Code")
    print("  → Copy the exact code from that page and run it")
print("=" * 60)