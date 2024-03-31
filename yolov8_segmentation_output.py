import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n-seg.pt')

# Path to the video file
video_path = r"cars.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Perform object detection and segmentation on the frame
        results = model(frame)

        # Iterate over detection results and filter out detections with confidence scores below 0.75
        filtered_results = []
        for result in results:
            filtered_boxes = [box for box, prob in zip(result.boxes[0], result.probs[0]) if prob >= 0.75]
            filtered_results.append(result.clone(boxes=[filtered_boxes]))

        # Plot the annotated frame for each filtered result
        for filtered_result in filtered_results:
            annotated_frame = filtered_result.render()[0]
            cv2.imshow("YOLOv8 Interference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
