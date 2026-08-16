import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    model_path = r"D:\Asphr\Yolo model\argus_yolo.onnx"
    video_path = r"D:\Asphr\Yolo model\testvideo.mp4"
    output_video_path = r"D:\Asphr\Yolo model\output_detected.mp4"

    print("==================================================")
    print("      ASPHR YOLO HAZARD INFERENCE DETECTOR        ")
    print("==================================================")
    print(f"Loading ONNX YOLO Model: {model_path}")
    print(f"Input Video File:      {video_path}")

    # Load YOLO model
    try:
        model = YOLO(model_path, task='detect')
        print(f"Model loaded successfully. Model Names/Classes: {model.names}")
    except Exception as e:
        print(f"Failed to load ONNX model via Ultralytics: {e}")
        return

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open input video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video Info: {width}x{height} @ {fps:.2f} FPS | Total Frames: {total_frames}")

    # Define Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    total_detections_count = 0
    class_detection_counts = {}
    frame_detection_log = []

    print("\nRunning Frame-by-Frame Inference...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run YOLO inference on frame
        results = model.predict(source=frame, conf=0.25, verbose=False)

        frame_detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()

                class_name = model.names.get(cls_id, f"Class {cls_id}")

                frame_detections.append({
                    "class_name": class_name,
                    "confidence": conf,
                    "box": [round(c, 1) for c in xyxy]
                })

                # Update global stats
                class_detection_counts[class_name] = class_detection_counts.get(class_name, 0) + 1
                total_detections_count += 1

                # Draw bounding box and label on annotated frame
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Color code by class type
                if "pothole" in class_name.lower():
                    color = (0, 0, 255) # Red for potholes
                elif "pedestrian" in class_name.lower() or "person" in class_name.lower():
                    color = (255, 165, 0) # Orange for pedestrians
                else:
                    color = (0, 255, 255) # Yellow for obstacles

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write overlay banner
        timestamp_sec = frame_count / fps
        banner_text = f"Frame: {frame_count}/{total_frames} ({timestamp_sec:.1f}s) | Detections: {len(frame_detections)}"
        cv2.putText(frame, banner_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)

        if len(frame_detections) > 0:
            frame_detection_log.append({
                "frame": frame_count,
                "timestamp_sec": round(timestamp_sec, 2),
                "detections": frame_detections
            })

        if frame_count % 30 == 0 or frame_count == total_frames:
            print(f"Processed Frame {frame_count}/{total_frames} ({timestamp_sec:.1f}s) - {len(frame_detections)} objects detected")

    cap.release()
    out.release()

    print("\n==================================================")
    print("           INFERENCE RESULTS SUMMARY              ")
    print("==================================================")
    print(f"Processed Frames:      {frame_count}/{total_frames}")
    print(f"Total Detected Objects: {total_detections_count}")
    print("\nClass Breakdown:")
    for cls_name, count in class_detection_counts.items():
        print(f"  - {cls_name}: {count} detections")

    print("\nSample Detections Log (Key Frames):")
    for log_item in frame_detection_log[:15]:
        det_summary = ", ".join([f"{d['class_name']} ({d['confidence']:.2f})" for d in log_item['detections']])
        print(f"  Frame {log_item['frame']} ({log_item['timestamp_sec']}s): {det_summary}")

    print(f"\nAnnotated Video Saved To: {output_video_path}")

if __name__ == "__main__":
    main()
