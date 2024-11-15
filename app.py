from flask import Flask, render_template, Response
import cv2
import supervision as sv
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
import time
import requests

app = Flask(__name__)

model = YOLO('best (1).pt')  # โหลดโมเดล YOLO

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Create folder for saving images
save_folder = r"D:\petch\zoomed_detections"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Timing for saving images
last_save_time = 0

# LINE Notify API settings
line_notify_token = 'Fd4h4pEAHKCXatr8JU7x7hwkDIe9wmK6pzkqH5b8Ipv'  # แทนที่ด้วย LINE Notify token ของคุณ
line_notify_api = 'https://notify-api.line.me/api/notify'


def send_line_notify(message, image_path):
    headers = {
        'Authorization': f'Bearer {line_notify_token}'
    }
    payload = {
        'message': message
    }
    files = {
        'imageFile': open(image_path, 'rb')
    }
    r = requests.post(line_notify_api, headers=headers, data=payload, files=files)
    return r.status_code


def generate_annotated_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # MediaPipe Pose Detection
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(image_rgb)

        # YOLO Object Detection
        results = model(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        # Filter detections based on confidence score
        filtered_indices = confidences >= 0.4
        boxes = boxes[filtered_indices]
        confidences = confidences[filtered_indices]
        class_ids = class_ids[filtered_indices]

        # Annotate frame
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        labels = [f"{model.names[int(class_id)]}" for class_id in class_ids]
        detections = sv.Detections(
            xyxy=boxes,
            class_id=class_ids,
            confidence=confidences
        )

        annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, labels=labels, detections=detections)

        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_zoomed_frames():
    global last_save_time
    while True:
        success, frame = cap.read()
        if not success:
            break

        # MediaPipe Pose Detection
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(image_rgb)

        # YOLO Object Detection
        results = model(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        # Filter detections based on confidence score
        filtered_indices = confidences >= 0.4
        boxes = boxes[filtered_indices]
        confidences = confidences[filtered_indices]
        class_ids = class_ids[filtered_indices]

        # Check for person and smoke/vape detection (class ID for smoke = 0, and vape = 1)
        person_detected = results_pose.pose_landmarks is not None
        smoke_detected = any(class_id == 0 for class_id in class_ids)
        vape_detected = any(class_id == 1 for class_id in class_ids)

        if person_detected and (smoke_detected or vape_detected):
            # Calculate bounding box around person
            landmark_array = []
            for landmark in results_pose.pose_landmarks.landmark:
                landmark_array.append((int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])))

            x_min = min([pt[0] for pt in landmark_array])
            y_min = min([pt[1] for pt in landmark_array])
            x_max = max([pt[0] for pt in landmark_array])
            y_max = max([pt[1] for pt in landmark_array])

            # Ensure the bounding box is within the image dimensions
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(frame.shape[1], x_max)
            y_max = min(frame.shape[0], y_max)

            # Adjust offset to make the bounding box taller
            offset = 100  # Increase this value to make the bounding box taller
            y_min = max(0, y_min - offset)
            y_max = min(frame.shape[0], y_max + offset)

            if x_max > x_min and y_max > y_min:  # Ensure valid dimensions
                zoomed_frame = frame[y_min:y_max, x_min:x_max]
                ret, buffer = cv2.imencode('.jpg', zoomed_frame)
                frame = buffer.tobytes()

                # Save the image after 5 seconds
                current_time = time.time()
                if current_time - last_save_time >= 5:
                    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                    save_path = os.path.join(save_folder, f"zoomed_detection_{timestamp}.jpg")
                    cv2.imwrite(save_path, zoomed_frame)
                    last_save_time = current_time

                    # Send LINE Notify with the saved image
                    message = f"Detection alert at {timestamp}"
                    send_line_notify(message, save_path)

        else:
            # If no detection, create a black image
            frame = np.zeros_like(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed_annotated')
def video_feed_annotated():
    return Response(generate_annotated_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_zoomed')
def video_feed_zoomed():
    return Response(generate_zoomed_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
