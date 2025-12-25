import torch
import cv2
import numpy as np
from ultralytics import YOLO  # Ensure you have YOLOv10 installed

class YOLOv10Detector:
    def __init__(self, model_path):
        """Initialize YOLOv10 model."""
        self.model = YOLO(model_path)  # Load YOLOv10 model

    def image_detection(self, img_path):
        """Detect objects in the image and display only class names (Optimized)."""
        try:
            img = cv2.imread(img_path)  # Load the image
            if img is None:
                print("Error: Failed to load image.")
                return

            # Perform object detection (optimized)
            results = self.model.predict(img, verbose=False)  # Faster inference

            # Extract unique detected classes
            detected_classes = {self.model.names[int(obj.cls)] for obj in results[0].boxes}

            # Display results instantly
            if detected_classes:
                print("/nDetected Objects:")
                print("/n".join(f"- {class_name}" for class_name in detected_classes))
            else:
                print("Error")

        except Exception as e:
            print(f"Error during YOLO detection: {e}")

# Example usage
model_path = "D:/Mini_Project/object_detection/Notes/yolov10m.pt"  # Path to your YOLOv10 model
image_path = "C:/Users/laksh/OneDrive/Desktop/KneeXray-images/0Normal"  # Path to test image

detector = YOLOv10Detector(model_path)
detector.image_detection(image_path)
