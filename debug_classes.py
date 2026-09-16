"""Quick debug script to check what class IDs RF-DETR returns for the camera feed."""
import os, cv2, numpy as np
from rfdetr import RFDETRLarge
import torch

RTSP_URLS = [
    "rtsp://admin:123456@192.168.1.35:554/0",
    "rtsp://admin:123456@192.168.1.35:554/stream1",
    "rtsp://admin:123456@192.168.1.35:554/1",
]

print("Loading model...")
model = RFDETRLarge()
print(f"class_names: {model.class_names[:10]}")

print("\nConnecting to camera...")
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
cap = None
for url in RTSP_URLS:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if cap.isOpened():
        print(f"Connected: {url}")
        break
    cap.release()
    cap = None

if cap is None:
    print("Failed to connect!")
    exit(1)

# Read 5 frames and show all detections
for i in range(5):
    ret, frame = cap.read()
    if not ret:
        print(f"Frame {i}: no frame")
        continue
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = model.predict(rgb, threshold=0.15)
    
    print(f"\n--- Frame {i} (shape: {frame.shape}) ---")
    print(f"Total detections: {len(detections)}")
    
    if len(detections) > 0:
        unique_classes = set(detections.class_id.tolist())
        print(f"Unique class_ids found: {unique_classes}")
        for cls_id in sorted(unique_classes):
            mask = detections.class_id == cls_id
            count = mask.sum()
            name = model.class_names[cls_id] if cls_id < len(model.class_names) else f"unknown_{cls_id}"
            confs = detections.confidence[mask]
            print(f"  class_id={cls_id} ({name}): {count} detections, confidence range: [{confs.min():.3f}, {confs.max():.3f}]")
    else:
        print("  No detections at all!")

cap.release()
print("\nDone!")
