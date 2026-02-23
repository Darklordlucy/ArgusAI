import cv2
import numpy as np
import onnxruntime as ort
import time

if __name__ == "__main__":

    ONNX_PATH  = "D:/IrisAI/runs/detect/Argus_AI/model/runs/argus_ai_v16/weights/best.onnx"
    CLASSES    = ["pothole", "pedestrian", "obstacle"]
    CLS_CONF   = {"pothole": 0.20, "pedestrian": 0.10, "obstacle": 0.20}
    COLORS     = {"pothole": (0,0,255), "pedestrian": (0,165,255), "obstacle": (128,0,128)}

    session    = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print("Model loaded.")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Camera opened. Press Q to quit.")

    fps_list    = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0   = time.perf_counter()
        h, w = frame.shape[:2]

        # Preprocess
        blob = cv2.resize(frame, (640, 640))
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, :]

        # Inference
        dets = session.run(None, {input_name: blob})[0][0]   # (300, 6)

        # Every 30 frames print top 3 raw confidences (debug)
        frame_count += 1
        if frame_count % 30 == 0:
            top = sorted(dets, key=lambda x: x[4], reverse=True)[:3]
            for d in top:
                print(f"  cls={CLASSES[int(d[5])]}  conf={d[4]:.3f}")

        # Draw detections
        for det in dets:
            x1, y1, x2, y2, conf, cls_id = det
            cls_name = CLASSES[int(cls_id)]

            if conf < CLS_CONF[cls_name]:
                continue

            # Scale from 640 space to display size
            sx  = w / 640
            sy  = h / 640
            x1  = int(x1 * sx);  y1 = int(y1 * sy)
            x2  = int(x2 * sx);  y2 = int(y2 * sy)

            color = COLORS[cls_name]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}",
                        (x1, max(y1-8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # FPS
        fps = 1 / (time.perf_counter() - t0)
        fps_list.append(fps)
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Argus-AI", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Average FPS: {np.mean(fps_list):.1f}")