import torch
from ultralytics import YOLO

if __name__ == "__main__":
    torch.cuda.empty_cache()

    model = YOLO("D:/IrisAI/runs/detect/Argus_AI/model/runs/argus_ai_v16/weights/best.pt")
    model.export(format="onnx", imgsz=640, simplify=True)

    print("Export complete — best.onnx ready for ZenDNN benchmark.")