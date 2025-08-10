import cv2
import subprocess
import torch # Not needed for YOLOv8 but kept for potential future use
import numpy as np
import pandas as pd
import time
import os
from ultralytics import YOLO  # YOLOv8

# Configuration
youtube_url = "https://www.youtube.com/watch?v=86-7Dr7yeVQ"
model_path = "yolov8m.pt"  # Medium model for better accuracy
csv_file = "detections_log.csv"

model_conf_threshold = 0.25   # YOLO min confidence for detection
min_csv_conf = 0.85           # Confidence threshold for logging
frame_skip = 10               # Process every Nth frame
max_logged_lines = 500        # Max lines in CSV
run_seconds = 120             # Max runtime in seconds
width, height = 1920, 1080    # Expected stream resolution
frame_delay = 0.05            # Artificial delay to control FPS

duplicate_reset_seconds = 5   # How long before duplicates can be logged again

# Loading YOLOv8 model
print("[INFO] Loading YOLOv8 model...")
model = YOLO(model_path)
model.conf = model_conf_threshold

# Prepare CSV file if it doesn’t exist
if not os.path.exists(csv_file):
    pd.DataFrame(columns=["timestamp", "class", "confidence", "x_min", "y_min", "x_max", "y_max"]) \
        .to_csv(csv_file, index=False)

# Start YouTube livestream using streamlink and ffmpeg
print("[INFO] Starting YouTube livestream at 1080p...")
ffmpeg_cmd = (
    f"streamlink {youtube_url} 1080p --stdout | "
    f"ffmpeg -i pipe:0 -f rawvideo -pix_fmt bgr24 -"
)

process = subprocess.Popen(
    ffmpeg_cmd,
    shell=True,
    stdout=subprocess.PIPE,
    bufsize=10**8
)

# Process video frames
start_time = time.time()
frame_count = 0
logged_lines = 0

# To track recent detections and avoid duplicates
recent_detections = set()
last_duplicate_reset = time.time()

# Main loop
try:
    while True:
        # Check for runtime limit
        if time.time() - start_time > run_seconds:
            print("[INFO] Run time limit reached, exiting...")
            break
        
        # Read raw frame
        raw_frame = process.stdout.read(width * height * 3)
        if not raw_frame:
            print("[INFO] No more frames, exiting...")
            break
        
        # Helps with frame skipping
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Convert raw frame to numpy array
        frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))

        # Perform detection
        results = model(frame, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()

        # Reset recent detections periodically
        if time.time() - last_duplicate_reset > duplicate_reset_seconds:
            recent_detections.clear()
            last_duplicate_reset = time.time()

        # Parse detections
        for det in detections:
            x_min, y_min, x_max, y_max, conf, cls_id = det
            class_name = model.names[int(cls_id)]
            if conf >= min_csv_conf:
                det_key = (
                    class_name,
                    round(x_min, -1),
                    round(y_min, -1),
                    round(x_max, -1),
                    round(y_max, -1)
                )

                # Log detection if not a recent duplicate
                if det_key not in recent_detections and logged_lines < max_logged_lines:
                    pd.DataFrame([{
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "class": class_name,
                        "confidence": float(conf),
                        "x_min": float(x_min),
                        "y_min": float(y_min),
                        "x_max": float(x_max),
                        "y_max": float(y_max)
                    }]).to_csv(csv_file, mode='a', header=False, index=False)

                    logged_lines += 1
                    recent_detections.add(det_key)

        # Render detections
        annotated_frame = results[0].plot()
        cv2.imshow("YouTube Stream - YOLOv8 Detection", annotated_frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] User requested exit.")
            break
        
        # Control frame rate
        time.sleep(frame_delay)

# Cleanup
finally:
    process.terminate()
    cv2.destroyAllWindows()
    print(f"[INFO] Detections saved to {csv_file}")