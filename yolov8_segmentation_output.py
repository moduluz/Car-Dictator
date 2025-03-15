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

# Run detection on the frame
results = model(frame)

# Process the results (just one result per frame in practice)
for result in results:
    # Filter boxes with confidence >= 0.75
    # Access confidence scores correctly
    boxes = result.boxes
    mask = boxes.conf >= 0.75
    filtered_boxes = boxes[mask]
    
    # Create a new result with only the filtered boxes
    # (This depends on how the YOLO implementation handles results)
    
    # Render the filtered results
    annotated_frame = result.plot()  # Method name might be different based on YOLO version
    cv2.imshow("YOLOv8 Inference", annotated_frame)
