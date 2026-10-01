import os
import cv2

URL = "rtsp://beumer:Beumer!123@172.20.45.131:554/video/live?channel=1&subtype=1"

OPTIONS = "rtsp_transport;tcp|stimeout;3000000|max_delay;50000|reorder_queue_size;0"

print("OpenCV version:", cv2.__version__)
print("FFmpeg options:", OPTIONS)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = OPTIONS

print("\nOpening RTSP with production FFmpeg options...")

cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)

print("isOpened:", cap.isOpened())

if cap.isOpened():
    ret, frame = cap.read()

    print("read:", ret)

    if frame is not None:
        print("frame shape:", frame.shape)

    cap.release()

print("\nTest complete.")