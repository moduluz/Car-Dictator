# CAR-DICTATOR {Vehicle Detector}


This repository contains a Python script for performing object detection and segmentation using YOLOv8 on a video file. It utilizes the `cv2` library for video processing and the `ultralytics` library for YOLOv8 model inference.

## Installation

1. Clone the repository:
   ```bash
   git clone hhttps://github.com/moduluz/Car-Dictator

2. Install the required dependencies:
   ```bash
   pip install opencv-python-headless
   pip install git+https://github.com/ultralytics/yolov5.git

3. Download the YOLOv8 model weights (yolov8n-seg.pt) from the Ultralytics YOLOv5 GitHub repository.


## Usage
1. Place the downloaded YOLOv8 model weights (yolov8n-seg.pt) in the same directory as the Python script.

2. Modify the video_path variable in the script to specify the path to the video file you want to process.

3. Run the script:
   ```bash
   python yolo_object_detection.py

4. Press the q key to exit the video stream.

## Notes
Ensure that you have OpenCV installed (cv2) as it's a dependency for video processing.

Adjust the confidence threshold (0.75 in the code) as needed to filter out detections with lower confidence scores.

This script is designed for educational and demonstration purposes and may require modifications for production use.
