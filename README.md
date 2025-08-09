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


This stage successfully demonstrated real-time object detection on a YouTube livestream with YOLOv5, forming the foundation for further enhancements.

---

## Day 3: Detection Logging & Analysis

### Overview  
On Day 3, the focus was to extend the real-time object detection pipeline by extracting detailed detection data from YOLOv5 outputs and logging it for post-analysis. This involved capturing detected object classes, confidence scores, bounding box coordinates, and timestamps for each frame processed from a YouTube livestream.

### Key Achievements  
- Integrated YOLOv5 inference with live YouTube stream frames to extract detection metadata in real time.  
- Implemented a CSV logging mechanism that records object class, confidence, bounding box coordinates, and timestamp for each detection above a configurable confidence threshold.  
- Enabled graceful run-time limits to automatically stop processing after a fixed duration (120 seconds), facilitating manageable data sizes.  
- Supported user interrupt via keyboard (pressing 'q') to terminate streaming and save logged data.  
- Added frame skipping to reduce computational load and control the volume of logged data, improving performance and usability.  
- Introduced configurable parameters for minimum confidence to log detections and a maximum line limit to prevent excessively large CSV files.

### Implementation Details  
- Utilized `streamlink` piped to `ffmpeg` for robust and efficient frame extraction from the YouTube livestream at 1080p resolution.  
- Applied OpenCV to decode raw video frames from the subprocess stream for YOLOv5 model inference.  
- Leveraged PyTorch Hub to load a custom YOLOv5 pretrained model and set inference confidence thresholds.  
- Converted YOLOv5 predictions to pandas DataFrames for easy filtering and CSV writing.  
- Managed CSV file creation with proper headers and incremental appending of detection data in real time.  
- Implemented a frame counter to process every 10th frame, balancing detection frequency with system resource constraints.

### Challenges & Resolutions  
- **Excessive Logging Volume:** Initial implementation logged detections on every frame, resulting in thousands of entries within seconds, making the CSV file unwieldy.  
  - *Solution:* Added frame skipping (processing every 10th frame) and increased the minimum confidence threshold for logging detections to 0.7.  
  - Introduced a maximum logged lines cap (500) to prevent uncontrolled growth of the CSV file.  
- **Streaming Pipeline Complexity:** Handling live video streams from YouTube required a combined shell pipeline (`streamlink | ffmpeg`) executed via `subprocess.Popen` with `shell=True` to correctly pipe the raw video data into OpenCV.  
  - *Solution:* Maintained the pipeline as a single string command passed to the shell, ensuring compatibility and stream integrity.  
- **Real-time Processing Performance:** Processing 1080p livestream frames and running YOLOv5 inference in real time posed performance challenges, risking dropped frames or lag.  
  - *Solution:* Implemented frame skipping and tuned confidence thresholds to reduce inference frequency and computational overhead, preserving stream smoothness and detection relevance.  
- **Graceful Termination:** Ensured that both automatic run-time limit and manual user quit commands correctly close streams and save logged data without corrupting the CSV file or leaving dangling processes.  

### Next Steps  
- Further optimize detection speed and accuracy (Day 4 focus).  
- Explore smarter logging strategies such as deduplication of detections within short time windows.  
- Enhance analysis tools to visualize and summarize the logged detection data.


*This day’s work laid a solid foundation for robust, efficient, and manageable real-time object detection logging — a critical step towards building a reliable livestream analytics system.*
