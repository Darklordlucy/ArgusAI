import yaml
from pathlib import Path

BASE = Path("Argus_AI/data/raw_downloads")

for folder in sorted(BASE.iterdir()):
    if not folder.is_dir():
        continue

    # Find the yaml file (could be data.yaml or dataset.yaml)
    yaml_files = list(folder.glob("*.yaml")) + \
                 list(folder.glob("**/*.yaml"))

    print(f"\n{'─'*50}")
    print(f"Dataset: {folder.name}")

    if not yaml_files:
        print("  No YAML found")
        continue

    for yf in yaml_files[:1]:  # Only show first yaml
        try:
            with open(yf) as f:
                data = yaml.safe_load(f)
            print(f"  YAML:    {yf.name}")
            print(f"  Classes: {data.get('names', 'not found')}")
            print(f"  Count:   {data.get('nc', '?')} classes")

            # Count images
            for split in ["train", "valid", "val", "test"]:
                img_dir = folder / split / "images"
                if img_dir.exists():
                    imgs = list(img_dir.glob("*.jpg")) + \
                           list(img_dir.glob("*.png"))
                    print(f"  {split:6s}: {len(imgs)} images")
        except Exception as e:
            print(f"  Error reading YAML: {e}")