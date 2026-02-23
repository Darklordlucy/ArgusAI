# Argus_AI/scripts/validate.py
import torch
from ultralytics import YOLO

if __name__ == "__main__":
    torch.cuda.empty_cache()

    model   = YOLO("D:/IrisAI/runs/detect/Argus_AI/model/runs/argus_ai_v16/weights/best.pt")
    metrics = model.val(
        data    = "D:/ArgusAI/Argus_AI/data/dataset.yaml",
        batch   = 4,
        device  = "cuda",
        workers = 0,
    )

    print(f"mAP@0.5:      {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
    print(f"Precision:    {metrics.box.p.mean():.3f}")
    print(f"Recall:       {metrics.box.r.mean():.3f}")