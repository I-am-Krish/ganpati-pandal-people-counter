"""
Ganesh Pandal People Counter - Web Version
==========================================
Real-time people counting using RF-DETR + ByteTrack + supervision LineZone.
Serves a video stream over HTTP on port 7001.
"""

import os
import sys
import time
import threading
import json

import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRLarge
import torch
from flask import Flask, Response

app = Flask(__name__)

# ─── CONFIGURATION ───────────────────────────────────────────────────────────

RTSP_URLS = [
    "rtsp://admin:123456@192.168.1.35:554/0",
    "rtsp://admin:123456@192.168.1.35:554/stream1",
    "rtsp://admin:123456@192.168.1.35:554/1",
    "rtsp://admin:123456@192.168.1.35:554/Streaming/Channels/101",
    "rtsp://admin:123456@192.168.1.35:554/cam/realmonitor?channel=1&subtype=0",
]

CONFIDENCE_THRESHOLD = 0.35
TRACK_ACTIVATION_THRESHOLD = 0.25
LOST_TRACK_BUFFER = 60
MINIMUM_MATCHING_THRESHOLD = 0.8
FRAME_RATE = 15

# Colors
COLOR_HEADER = (50, 50, 50)
COLOR_IN = (0, 255, 0)
COLOR_OUT = (0, 0, 255)

# Globals for sharing between threads if needed
current_frame = None

# ─── THREADED FRAME GRABBER ──────────────────────────────────────────────────

class RTSPFrameGrabber:
    def __init__(self, source):
        self.source = source
        self.cap = None
        self.frame = None
        self.ret = False
        self.lock = threading.Lock()
        self.stopped = False
        self.frame_count = 0

    def start(self):
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            return False
        self.ret, self.frame = self.cap.read()
        if not self.ret:
            return False
        thread = threading.Thread(target=self._update, daemon=True)
        thread.start()
        return True

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
                self.frame_count += 1
            if not ret:
                self.stopped = True
                break

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        if self.cap is not None:
            self.cap.release()

# ─── DISPLAY HELPERS ─────────────────────────────────────────────────────────

def draw_counter_overlay(frame, in_count, out_count, fps, tracker_count):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), COLOR_HEADER, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "GANESH PANDAL PEOPLE COUNTER", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    total = in_count + out_count
    cv2.putText(frame, f"IN: {in_count}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_IN, 2)
    cv2.putText(frame, f"OUT: {out_count}", (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_OUT, 2)
    cv2.putText(frame, f"TOTAL: {total}", (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Tracking: {tracker_count}", (15, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    fps_text = f"FPS: {fps:.1f}"
    text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.putText(frame, fps_text, (w - text_size[0] - 15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

def draw_tracking_annotations(frame, detections):
    if detections is None or len(detections) == 0:
        return frame
    box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.from_hex(["#00BFFF"]), thickness=2)
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(["#00BFFF"]),
        text_color=sv.Color.WHITE,
        text_scale=0.4,
        text_thickness=1,
        text_padding=3
    )
    labels = []
    if detections.tracker_id is not None:
        for tracker_id, class_id, confidence in zip(detections.tracker_id, detections.class_id, detections.confidence):
            labels.append(f"#{tracker_id} cls:{class_id} {confidence:.2f}")
    else:
        for class_id, confidence in zip(detections.class_id, detections.confidence):
            labels.append(f"cls:{class_id} {confidence:.2f}")
    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
    return frame

from dual_line_counter import SingleLineCounter

def draw_count_line(frame, line):
    """Draw the single counting line on the frame - bright cyan."""
    cv2.line(frame, (int(line[0][0]), int(line[0][1])), (int(line[1][0]), int(line[1][1])), (0, 255, 255), 3)
    cv2.putText(frame, "COUNTING LINE", (10, int(line[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame

# ─── INFERENCE GENERATOR ─────────────────────────────────────────────────────

def generate_frames():
    global current_frame
    print("\n[1/3] Loading RF-DETR Large model...")
    try:
        model = RFDETRLarge()
        if hasattr(model, 'inference'):
            model.inference(dtype=torch.float16)
            print("Model optimized for FP16 inference!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\n[2/3] Connecting to camera...")
    grabber = None
    for url in RTSP_URLS:
        grabber = RTSPFrameGrabber(url)
        if grabber.start():
            print(f"Connected to {url}")
            break
        grabber.stop()
        grabber = None
    
    if grabber is None:
        print("Could not connect to camera.")
        return

    ret, first_frame = grabber.read()
    if not ret:
        return
    h, w = first_frame.shape[:2]

    # Single counting line at the middle of the frame
    # Direction: Right-to-Left so that crossing Top->Bottom = IN, Bottom->Top = OUT
    base_y = h // 2
    line_start, line_end = (w, base_y), (0, base_y)
    
    if os.path.exists("line_config.json"):
        with open("line_config.json", "r") as f:
            cfg = json.load(f)
            line_start = (cfg["start"]["x"], cfg["start"]["y"])
            line_end = (cfg["end"]["x"], cfg["end"]["y"])

    print("\n[3/3] Starting tracking loop...")
    
    # Using the global tracking variables which are tuned to be very forgiving
    tracker = sv.ByteTrack(
        track_activation_threshold=TRACK_ACTIVATION_THRESHOLD,
        lost_track_buffer=LOST_TRACK_BUFFER,
        minimum_matching_threshold=MINIMUM_MATCHING_THRESHOLD,
        frame_rate=FRAME_RATE,
    )

    counter = SingleLineCounter(line_start, line_end, dead_zone=5, cooldown_frames=30, timeout_seconds=10.0)
    print(f"[LINES] Counting line: {line_start} -> {line_end}")
    print(f"[LINES] Frame size: {w}x{h}, base_y={base_y}")

    fps = 0.0
    fps_start_time = time.time()
    fps_frame_count = 0

    try:
        while True:
            ret, frame = grabber.read()
            if not ret or frame is None:
                time.sleep(0.1)
                continue

            # Inference with RF-DETR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = model.predict(rgb_frame, threshold=CONFIDENCE_THRESHOLD)
            
            # The overhead angle frequently causes the model to misclassify people's heads/shoulders
            # as bicycles (1) or motorcycles (3). If we don't include them, the tracker drops people.
            person_mask = np.isin(detections.class_id, [0, 1, 3])
            detections = detections[person_mask]
            
            # Force all accepted detections to be strictly identified as Class 0 (Human)
            if len(detections) > 0:
                detections.class_id[:] = 0

            # Tracking
            detections = tracker.update_with_detections(detections)

            # State Machine Counting
            counter.update(detections)

            # Debug: log every 30 frames
            fps_frame_count += 1
            if fps_frame_count % 30 == 0:
                print(f"[DEBUG] Frame {fps_frame_count}: {len(detections)} tracked, IN={counter.entered_count}, OUT={counter.exited_count}, active_tracks={len(counter.tracks)}")

            # FPS calculation
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start_time = time.time()

            # Annotations
            annotated = frame.copy()
            annotated = draw_tracking_annotations(annotated, detections)
            annotated = draw_count_line(annotated, counter.line)
            annotated = draw_counter_overlay(annotated, counter.entered_count, counter.exited_count, fps, len(detections))

            # Encode for web
            ret, buffer = cv2.imencode('.jpg', annotated)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        print("[INFO] Client disconnected, releasing camera connection.")
        if grabber is not None:
            grabber.stop()

# ─── FLASK APP ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>Ganesh Pandal People Counter</title>
            <style>
                body { background-color: #121212; color: white; text-align: center; font-family: sans-serif; }
                img { max-width: 100%; height: auto; border: 2px solid #333; margin-top: 20px; }
            </style>
        </head>
        <body>
            <h1>Live Counter Feed</h1>
            <p>Tracking 50-60+ people using RF-DETR Large (Optimized)</p>
            <img src="/video_feed" />
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("Starting Web Server on port 7001...")
    # Run Flask server
    app.run(host='0.0.0.0', port=7001, threaded=True)
