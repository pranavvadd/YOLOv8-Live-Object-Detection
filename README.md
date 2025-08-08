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

