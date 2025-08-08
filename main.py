import cv2
import subprocess

youtube_url = "https://www.youtube.com/watch?v=rnXIjl_Rzy4"

# Use streamlink to fetch the YouTube stream and pipe it to OpenCV
streamlink_cmd = [
    "streamlink",
    "--stdout",
    youtube_url,
    "best"
]

# Start the streamlink process
process = subprocess.Popen(streamlink_cmd, stdout=subprocess.PIPE)

# OpenCV to read from the stream
cap = cv2.VideoCapture(process.stdout)

# Check if the stream opened successfully
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('YouTube Stream', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
process.kill()
cv2.destroyAllWindows()