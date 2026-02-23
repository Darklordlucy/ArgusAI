import torch
from ultralytics import YOLO
from tqdm import tqdm

# ── Must be at module level for Windows multiprocessing pickling ──────────────
bar = None

def on_train_epoch_end(trainer):
    bar.set_postfix({
        "box_loss": f"{trainer.loss_items[0]:.3f}",
        "cls_loss": f"{trainer.loss_items[1]:.3f}",
        "mAP50":    f"{trainer.metrics.get('metrics/mAP50(B)', 0):.3f}",
    })
    bar.update(1)


def main():
    global bar

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    if device == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Model
    model  = YOLO("yolov10n.pt")
    epochs = 50
    bar    = tqdm(total=epochs, desc="Training", unit="epoch")
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    # Train
    model.train(
        data     = "D:/ArgusAI/Argus_AI/data/dataset.yaml",
        epochs   = epochs,
        batch    = 8,
        imgsz    = 640,
        device   = device,
        workers  = 2,        # ← changed from 2 to 0 (no subprocess spawning)
        cache    = False,    # ← changed from True (70GB needed, impossible)
        project  = "Argus_AI/model/runs",
        name     = "argus_ai_v1",
        patience = 20,
        verbose  = False,
    )

    bar.close()
    print(f"Training complete. Device used: {device}")

    # Validate
    metrics = model.val(data="D:/ArgusAI/Argus_AI/data/dataset.yaml")
    print(f"mAP@0.5:      {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
    print(f"Precision:    {metrics.box.p.mean():.3f}")
    print(f"Recall:       {metrics.box.r.mean():.3f}")

    # Export
    model.export(format="onnx", imgsz=640, simplify=True)
    print("Export complete.")


if __name__ == "__main__":
    main()