import cv2
import subprocess
import torch
import numpy as np
import pandas as pd
import time
import os

# Config
youtube_url = "https://www.youtube.com/watch?v=86-7Dr7yeVQ"
model_path = "yolov5s.pt"

# Parameters
model_conf_threshold = 0.25      # YOLO min confidence to consider detection
min_csv_conf = 0.7               # Min confidence to log detection
frame_skip = 10                  # Process every 10th frame
max_logged_lines = 500           # Max lines to log in CSV
run_seconds = 120                # Max run time
width, height = 1920, 1080       # Frame size for 1080p

csv_file = "detections_log.csv"

# Ensure the model path exists
print("[INFO] Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.conf = model_conf_threshold

# Ensure the CSV file exists
if not os.path.exists(csv_file):
    pd.DataFrame(columns=["timestamp", "class", "confidence", "x_min", "y_min", "x_max", "y_max"]).to_csv(csv_file, index=False)

# Clear the CSV file if it already exists
print("[INFO] Starting YouTube livestream at 1080p...")
ffmpeg_cmd = (
    f"streamlink {youtube_url} 1080p --stdout | "
    f"ffmpeg -i pipe:0 -f rawvideo -pix_fmt bgr24 -"
)

# Start the FFmpeg process to read the YouTube stream
process = subprocess.Popen(
    ffmpeg_cmd,
    shell=True,
    stdout=subprocess.PIPE,
    bufsize=10**8
)

# Check if the process started successfully
start_time = time.time()
frame_count = 0
logged_lines = 0

# Main loop to read frames and process detections
while True:
    # Check if the run time limit has been reached
    if time.time() - start_time > run_seconds:
        print("[INFO] Run time limit reached, exiting...")
        break
    # Read raw frame data from the FFmpeg process
    raw_frame = process.stdout.read(width * height * 3)
    if not raw_frame:
        print("[INFO] No more frames, exiting...")
        break
     # Check if the frame size matches the expected size
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue
     # Convert raw frame data to a NumPy array
    frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))

    # Resize the frame to match the model input size
    results = model(frame)
    detections = results.pandas().xyxy[0]

    # Log detections to CSV if they meet the confidence threshold
    for _, row in detections.iterrows():
        if row['confidence'] >= min_csv_conf:
            if logged_lines >= max_logged_lines:
                print("[INFO] Max logged lines reached, stopping logging but continuing video...")
                break
             # Log the detection to the CSV file
            pd.DataFrame([{
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "class": row['name'],
                "confidence": float(row['confidence']),
                "x_min": row['xmin'],
                "y_min": row['ymin'],
                "x_max": row['xmax'],
                "y_max": row['ymax']
            }]).to_csv(csv_file, mode='a', header=False, index=False)
            logged_lines += 1

    # Display the frame with detections
    annotated_frame = np.squeeze(results.render())
    cv2.imshow("YouTube Stream - YOLOv5 Detection", annotated_frame)

    # Check for user exit request
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] User requested exit.")
        break

# Cleanup
process.terminate()
cv2.destroyAllWindows()
print(f"[INFO] Detections saved to {csv_file}")