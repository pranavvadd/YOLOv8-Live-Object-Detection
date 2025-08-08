import cv2
import subprocess
import torch
import numpy as np

# Config
youtube_url = "https://www.youtube.com/watch?v=rnXIjl_Rzy4"
confidence_threshold = 0.25
model_path = "yolov5s.pt"
width, height = 1920, 1080  # Force 1080p

# Load YOLOv5
print("[INFO] Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.conf = confidence_threshold

# Start streamlink -> ffmpeg pipeline
print("[INFO] Starting YouTube livestream at 1080p...")
ffmpeg_cmd = (
    f"streamlink {youtube_url} 1080p --stdout | "
    f"ffmpeg -i pipe:0 -f rawvideo -pix_fmt bgr24 -"
)

# Run process
process = subprocess.Popen(
    ffmpeg_cmd,
    shell=True,
    stdout=subprocess.PIPE,
    bufsize=10**8
)

while True:
    raw_frame = process.stdout.read(width * height * 3)
    if not raw_frame:
        break

    frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))

    # YOLO inference
    results = model(frame)
    annotated_frame = np.squeeze(results.render())

    # Show annotated stream
    cv2.imshow("YouTube Stream - YOLOv5 Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

process.terminate()
cv2.destroyAllWindows()