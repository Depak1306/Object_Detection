# Object Detection using YOLO

## Overview
This project implements an object detection application using a YOLO-based deep learning model.
It detects predefined object classes from images, videos, and real-time webcam streams.

## Project Structure
- app.py – Main application
- requirements.txt – Dependencies

## Installation
pip install -r requirements.txt

## Usage
python app.py

## Model Weights
Pretrained YOLO model weights are not included in this repository.
Please download them separately using Ultralytics and update the model path in app.py.

### Download Weights Example
```bash
pip install ultralytics
yolo predict model=yolov8n.pt 
