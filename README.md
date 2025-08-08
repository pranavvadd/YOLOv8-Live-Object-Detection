# Real-Time YouTube Livestream Object Detection with YOLOv5

## Day 1: Environment Setup & Initial Tests

### Overview
On Day 1, the goal was to set up the development environment, verify YOLOv5 installation, and confirm the ability to capture and display frames from a YouTube livestream. This foundational step ensures the system can process live video input, which is critical for real-time object detection.

### Key Tasks Completed
- Installed Python 3.13 and created a virtual environment to isolate dependencies.
- Installed essential libraries: PyTorch, OpenCV, Streamlink, and YOLOv5 dependencies.
- Cloned the YOLOv5 repository and successfully ran inference on a static image (`bus.jpg`) using pretrained weights.
- Tested capturing frames from a YouTube livestream using Streamlink piped directly into OpenCV for display.

### Challenges and Solutions
- **Issue:** YOLOv5 Git integration produced a warning related to deprecated `pkg_resources` and a “cannot change directory” error due to spaces in file paths.  
  **Solution:** This warning was non-blocking. To avoid future confusion, project directories with spaces will be renamed.

- **Issue:** OpenCV attempted to access the system webcam when trying to read livestream frames, leading to macOS permission errors and warnings about deprecated camera APIs.  
  **Solution:** Adjusted code to avoid webcam initialization by piping Streamlink output directly into OpenCV via `cv2.VideoCapture(process.stdout)`. This circumvents macOS’s camera permission system and aligns with livestream processing requirements.

- **Issue:** Streamlink defaulted to opening the VLC media player, which was not installed, causing errors in the terminal.  
  **Solution:** Added the `--stdout` flag to Streamlink commands, enabling direct piping of livestream data into Python without an external player.

### Next Steps
- Integrate YOLOv5 model inference with the live video feed from the YouTube livestream.
- Optimize frame processing for real-time detection speed.
- Implement detection logging and analysis.

---

## Day 2: Livestream Frame Processing & YOLOv5 Integration

### Objectives
- Stream video frames from a YouTube livestream using Streamlink and FFmpeg
- Feed livestream frames into the YOLOv5 model for real-time object detection
- Render and display detected bounding boxes, class labels, and confidence scores on the live video
- Lock stream quality at 1080p to improve detection clarity and reduce pixelation issues

### Implementation Details

- Used `streamlink` to fetch the YouTube livestream at **1080p** quality (`streamlink <url> 1080p --stdout`)
- Piped stream through `ffmpeg` to convert into raw video frames with pixel format `bgr24` compatible with OpenCV
- Captured raw frames in Python, converted to numpy arrays, and reshaped to (1080, 1920, 3) resolution
- Loaded YOLOv5 pretrained model (`yolov5s.pt`) via PyTorch Hub, setting a confidence threshold of 0.25
- Performed inference on each frame and rendered detection results (bounding boxes, class labels, confidence scores)
- Displayed annotated frames using OpenCV in a window titled “YouTube Stream - YOLOv5 Detection”
- Pressing `q` cleanly terminates the stream and closes all windows

### Observations & Challenges

- Initial stream frames appeared pixelated and blurry, typical for livestream start — locking at 1080p fixed this issue
- Model loading takes several seconds initially due to downloading weights and model setup
- Detection confidence scores were modest, and only a few objects (mainly people) were tagged — expected given streaming conditions and model size
- Pressing `q` to quit sometimes raised a `Broken pipe` error caused by abrupt closing of FFmpeg/Streamlink subprocess pipes; this is harmless and does not affect functionality

### Next Steps

- Tune confidence thresholds and experiment with larger YOLOv5 models (e.g., `yolov5m`, `yolov5l`) to improve detection accuracy
- Optimize frame processing speed to reduce latency and increase frame rate
- Develop detection logging and analysis (Day 3) for meaningful data extraction from live detections

---

This stage successfully demonstrated real-time object detection on a YouTube livestream with YOLOv5, forming the foundation for further enhancements.
