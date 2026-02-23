import numpy as np
import onnxruntime as ort

ONNX_PATH  = "D:/IrisAI/runs/detect/Argus_AI/model/runs/argus_ai_v16/weights/best.onnx"

session    = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
dummy      = np.random.rand(1, 3, 640, 640).astype(np.float32)
outputs    = session.run(None, {input_name: dummy})

print(f"Output shape: {outputs[0].shape}")
print(f"Sample detection: {outputs[0][0][0]}")