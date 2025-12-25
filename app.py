import cv2
import numpy as np
from PyQt5 import QtWidgets as qtw,QtGui
from PyQt5 import QtGui as qtg
from PyQt5.QtCore import *
from ultralytics import YOLO
from PyQt5 import QtWidgets as qtw
from PyQt5.QtCore import Qt
import threading 
from target_classes import target_classes as tc
class NewWindow(qtw.QWidget):
    def __init__(self, window_name):
        super().__init__()
        self.setWindowTitle(window_name)
        self.setAcceptDrops(True)  
        self.model=YOLO('yolov10m.pt')
        # Drag and Drop Label
        self.image_label = qtw.QLabel("Drag and drop an image/video here", self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #aaa; padding: 100px;")

        layout = qtw.QVBoxLayout(self)
        layout.addWidget(self.image_label)
        self.setLayout(layout)
        self.showNormal()

    # Handle drag enter event
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()  # Accept the drag if it's a file
        else:
            event.ignore()

    # Handle drop event
    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                # Get the file path and convert to lowercase
                file_path = url.toLocalFile().lower()

                # Check if it's an image file
                if url.isLocalFile() and file_path.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = url.toLocalFile()
                    self.image_detection(img_path)  
                    event.acceptProposedAction()

                # Check if it's a video file
                elif url.isLocalFile() and file_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = url.toLocalFile()
                    self.video_detection(video_path)  
                    event.acceptProposedAction()

                # Ignore if the file is not an image or video
                else:
                    event.ignore()


   
    def image_detection(self, img_path):
        try:
            model = self.model

            # Load the image
            img = cv2.imread(img_path)
            if img is None:
                print("Failed to load image.")
                return

            # Perform object detection
            results = model(img)

            # Function to generate a color for each class ID
            def generate_color(class_id):
                np.random.seed(class_id)  # Seed with class ID for consistent color
                return tuple(np.random.randint(0, 256, size=3).tolist())  # Random color in BGR format

            # Loop through the detected objects and filter only the target classes
            for result in results:
                for obj in result.boxes:
                    class_id = int(obj.cls)
                    class_name = model.names[class_id]
                    confidence_score = float(obj.conf)  # Convert to float for printing

                    if class_name in tc:
                        print(f"Detected {class_name} with confidence {confidence_score:.2f}")

                        # Get bounding box coordinates
                        x1, y1, x2, y2 = obj.xyxy[0].tolist()  # Convert to list and unpack
                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))  # Convert each coordinate to int

                        # Generate a color for the current class
                        color = generate_color(class_id)

                        # Draw bounding box and label for the matched classes
                        label = f"{class_name} {confidence_score:.2f}"
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Display the image with bounding boxes
            cv2.imshow("YOLOv8 Object Detection", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error during YOLO detection: {e}")
    

   
   
    def video_detection(self, video_path):
        try:
           
            model = self.model

            # Open the video file or start capturing from a webcam
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print("Error: Could not open video.")
                return

            # Function to generate a color for each class ID
            def generate_color(class_id):
                np.random.seed(class_id)  # Seed with class ID for consistent color
                return tuple(np.random.randint(0, 256, size=3).tolist())  # Random color in BGR format

            frame_count = 0
            skip_frames = 1  # Process every frame for better detection

            while True:
                # Read a frame from the video
                ret, frame = cap.read()
                if not ret:
                    print("Finished processing the video or no frame captured.")
                    break

                # Resize the frame for better detection and speed
                frame = cv2.resize(frame, (1280, 720))  # Adjust resolution as needed

                # Process every frame
                if frame_count % skip_frames == 0:
                    # Perform object detection on the current frame
                    results = model(frame, conf=0.2)  # Set a lower confidence threshold

                    # Loop through the detected objects and filter only the target classes
                    for result in results:
                        for obj in result.boxes:
                            class_id = int(obj.cls)
                            class_name = model.names[class_id]
                            confidence_score = float(obj.conf)

                            if class_name in tc:
                                # Get bounding box coordinates
                                x1, y1, x2, y2 = obj.xyxy[0].tolist()
                                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                                # Generate a color for the current class
                                color = generate_color(class_id)

                                # Draw bounding box and label for the matched classes
                                label = f"{class_name} {confidence_score:.2f}"
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Display the frame with bounding boxes
                cv2.imshow("YOLOv8 Object Detection", frame)

                # Increment frame count
                frame_count += 1

                # Press 'q' to exit early
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release the video capture object and close the display window
            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error occurred: {e}")




class WebcamWindow(qtw.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WebCam Detection")
        self.model = YOLO('yolov10m.pt')
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            exit()

        self.setGeometry(100, 100, 800, 600)

        # QLabel for video display
        self.video_label = qtw.QLabel(self)
        self.video_label.setGeometry(0, 0, 800, 600)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)  # Fast refresh for video feed

        self.running = True

    def generate_color(self, class_id):
        np.random.seed(class_id)
        return tuple(np.random.randint(0, 256, size=3).tolist())

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            self.timer.stop()
            self.cap.release()
            cv2.destroyAllWindows()
            return
        
        # Resize frame for better processing speed
        frame = cv2.resize(frame, (640, 480))  

        # Perform detection
        self.detect_objects(frame)

        # Convert frame to QImage and display
        height, width, channels = frame.shape
        bytes_per_line = channels * width
        q_image = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format_BGR888)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(q_image))

    def detect_objects(self, frame):
        try:
            results = self.model(frame, conf=0.3)  # Adjust confidence threshold if necessary

            for result in results:
                for obj in result.boxes:
                    x1, y1, x2, y2 = map(int, obj.xyxy[0])
                    class_index = int(obj.cls[0])
                    class_name = self.model.names[class_index]
                    confidence_score = float(obj.conf[0])

                    if class_name in tc:
                        color = self.generate_color(class_index)
                        label = f"{class_name} ({confidence_score:.2f})"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        print(f"Detected: {class_name} ({confidence_score:.2f})")  # Log detections

        except Exception as e:
            print(f"Error during YOLO detection: {e}")

    def closeEvent(self, event):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

class AnimatedButton(qtw.QPushButton): 
    def __init__(self, icon_path,text,window_name,*args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.original_geometry = None 
        self.setFixedSize(500,250)
        self.setStyleSheet("""background-color: #c4b5fb; border-radius:50px; border:2px solid black;""")
        #   Icon
        self.icon=qtg.QIcon(icon_path)
        self.setIcon(self.icon)
        self.setIconSize(QSize(200,200))
        #   set Text
        self.setText(text)
        self.setStyleSheet(self.styleSheet() +" font-size:25px;color: black;font-weight: bold; font-family:Times New Roman;")
        #   Shadow Effect
        self.shadow_effect = qtw.QGraphicsDropShadowEffect()
        self.shadow_effect.setBlurRadius(90)
        self.shadow_effect.setXOffset(5)
        self.shadow_effect.setYOffset(5)
        self.shadow_effect.setColor(qtg.QColor(0, 0, 0, 160)) 
        self.setGraphicsEffect(self.shadow_effect)
        self.shadow_effect.setEnabled(False)
        #Enter Event
        self.enter_animation = QPropertyAnimation(self, b"geometry")
        self.enter_animation.setDuration(300)  
        self.enter_animation.setEasingCurve(QEasingCurve.OutQuad)  

        #   Leave Event
        self.leave_animation = QPropertyAnimation(self, b"geometry")
        self.leave_animation.setDuration(300)  
        self.leave_animation.setEasingCurve(QEasingCurve.OutQuad)

        self.window_name=window_name
        self.clicked.connect(self.open_new_window)

    def enterEvent(self, event):  
        self.original_geometry = self.geometry()
        
        new_x = self.x() + 4
        new_rect = QRect(new_x, self.y(), self.width(), self.height())
        
        self.enter_animation.setStartValue(self.geometry())
        self.enter_animation.setEndValue(new_rect)
        self.enter_animation.start()
        self.shadow_effect.setEnabled(True)
        super().enterEvent(event)
    def leaveEvent(self, event):  
        # Animate back to the original size
        self.leave_animation.setStartValue(self.geometry())
        self.leave_animation.setEndValue(self.original_geometry)
        self.leave_animation.start()
        self.shadow_effect.setEnabled(False)
        super().leaveEvent(event)  

    # create a new window on each click
    def open_new_window(self):
        if self.window_name=='Image Detection' or self.window_name=='Video Detection':
            self.open_new_window=NewWindow(self.window_name)
            self.open_new_window.show()
        elif self.window_name=='Live Detection':
            self.open_new_window = WebcamWindow()
            self.open_new_window.show()
        
class MainWindow(qtw.QWidget):
    def __init__(self):
        super().__init__()
        #   window 
        self.setWindowTitle("Object Detection App")
        self.setWindowIcon(qtg.QIcon('D:/Mini_Project/object_detection/Notes/Images/object detection.png'))
        layout=qtw.QHBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(1,1,1,1)
        #   frame1
        self.frame1=qtw.QFrame(self)
        self.frame1.setStyleSheet("QFrame { background-color: #f1f0fb;background-image: url('D:/Mini_Project/object_detection/Notes/Images/loading_img.png'); background-repeat: no-repeat; background-position: center; }")
        layout.addWidget(self.frame1,3)
        #   frame2
        self.frame2=qtw.QFrame(self)
        self.frame2.setStyleSheet("background-color: #f1f0fb;")
        layout.addWidget(self.frame2,2)
        # frame2 subframe
        frame2_layout=qtw.QHBoxLayout(self.frame2)
        self.sub_frame=qtw.QFrame(self.frame2)
        self.sub_frame.setStyleSheet("""background-color: #f1f0fb; border-radius:40px; border: 2px solid darkblack; """)
        frame2_layout.addWidget(self.sub_frame)
        frame2_layout.setContentsMargins(40,30,40,30)
        #   subframe layout
        sub_frame_layout=qtw.QVBoxLayout(self.sub_frame)
        #   Image
        image_button=AnimatedButton('D:/Mini_Project/object_detection/Notes/Images/image.png','Image Detection','Image Detection')
        sub_frame_layout.addWidget(image_button,alignment=Qt.AlignCenter)

        #   Video button
        video_button=AnimatedButton('D:/Mini_Project/object_detection/Notes/Images/video.png','Video Detection','Video Detection')
        sub_frame_layout.addWidget(video_button, alignment=Qt.AlignCenter)
        #   webcam Detection
        webcam_button=AnimatedButton('D:/Mini_Project/object_detection/Notes/Images/webcam.png','capture','Live Detection')
        sub_frame_layout.addWidget(webcam_button,alignment=Qt.AlignCenter)
        sub_frame_layout.stretch(1)
        self.setLayout(layout)
        
        self.showMaximized()
app=qtw.QApplication([])
mw=MainWindow()
app.exec_()