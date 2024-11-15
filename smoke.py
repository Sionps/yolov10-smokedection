import cv2
import supervision as sv
from ultralytics import YOLO
import time
import os
from datetime import datetime
import requests
import mediapipe as mp

model = YOLO('best (1).pt')  # โมเดลที่ถูกฝึกมาแล้ว

# เริ่มต้นการใช้งานเว็บแคม
cap = cv2.VideoCapture(0)  # ใช้ 0 ถ้ามีเว็บแคมเพียงตัวเดียว

# ตรวจสอบว่าเว็บแคมเปิดได้ถูกต้องหรือไม่
if not cap.isOpened():
    print("เกิดข้อผิดพลาด: ไม่สามารถเปิดเว็บแคมได้")
    exit()

# สร้างตัวเขียนกรอบ
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# สร้างโฟลเดอร์สำหรับบันทึกรูปภาพ
save_folder = "zoomed_detections"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# ตั้งเวลาสำหรับบันทึกรูปภาพ
last_save_time = time.time()

# ตั้งค่า LINE Notify API
line_notify_token = ''  # ใส่ LINE Notify token ของคุณที่นี่
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


# เริ่มต้นใช้งาน MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

while True:
    # รับภาพจากเว็บแคมแบบเฟรมต่อเฟรม
    ret, frame = cap.read()

    if not ret:
        print("เกิดข้อผิดพลาด: ไม่สามารถอ่านเฟรมได้")
        break

    # แปลงภาพเป็น RGB เนื่องจาก MediaPipe ใช้ภาพ RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ทำการตรวจจับท่าทาง
    results_pose = pose.process(image_rgb)

    # ทำการตรวจจับวัตถุ
    results = model(frame)  # รับผลการทำนาย
    # เตรียมข้อมูลการตรวจจับและป้ายที่มีคะแนนความมั่นใจ
    boxes = results[0].boxes.xyxy.cpu().numpy()  # กรอบที่ตรวจจับได้
    confidences = results[0].boxes.conf.cpu().numpy()  # คะแนนความมั่นใจ
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # รหัสคลาสในรูปแบบตัวเลขเต็ม

    # สร้างป้าย
    labels = [f"{model.names[int(class_id)]}" for class_id in class_ids]  # ลบคะแนนความมั่นใจออกจากป้าย

    # สร้างวัตถุการตรวจจับ
    detections = sv.Detections(
        xyxy=boxes,
        class_id=class_ids,
        confidence=confidences
    )

    # วาดกรอบรอบวัตถุที่ตรวจจับได้ในเฟรม
    annotated_frame = box_annotator.annotate(
        scene=frame, detections=detections)

    # วาดป้ายในเฟรม
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, labels=labels, detections=detections)

    # ตรวจสอบการตรวจจับคนและการสูบบุหรี่
    person_detected = results_pose.pose_landmarks is not None  # ตรวจสอบว่ามีการตรวจจับจุดท่าทางหรือไม่
    smoke_detected = any(class_id == 0 for class_id in class_ids)  # รหัสคลาส 0 สำหรับการสูบบุหรี่
    vape_detected = any(class_id == 1 for class_id in class_ids)  # รหัสคลาส 1 สำหรับบุหรี่ไฟฟ้า

    # ตรวจสอบเงื่อนไขการทำงานของ Zoomed Detection
    if person_detected and smoke_detected and not vape_detected:
        # คำนวณกรอบรอบตัวคน
        landmark_array = []
        for landmark in results_pose.pose_landmarks.landmark:
            landmark_array.append((int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])))

        # รับพิกัดของกรอบ
        x_min = min([pt[0] for pt in landmark_array])
        y_min = min([pt[1] for pt in landmark_array])
        x_max = max([pt[0] for pt in landmark_array])
        y_max = max([pt[1] for pt in landmark_array])

        # วาดกรอบรอบตัวคน
        cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # สีเขียว

        # ย้ายกรอบขึ้นไปอยู่เหนือศีรษะ
        offset = 50  # ปรับค่านี้เพื่อกำหนดว่าต้องการให้กรอบอยู่เหนือศีรษะเท่าใด
        y_min = max(0, y_min - offset)  # ย้ายขอบด้านบนขึ้นไปโดย 'offset' พิกเซล
        y_max = y_min + (y_max - y_min)  # ปรับ y_max เพื่อรักษาความสูงของกรอบเดิม

        # วาดกรอบเหนือศีรษะของคน
        cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # สีน้ำเงิน

        # สร้างเฟรมขยายที่แสดงกรอบรวม
        zoomed_frame = frame[y_min:y_max, x_min:x_max]

        # ตรวจสอบขนาดของ zoomed_frame ก่อนแสดงผล
        if zoomed_frame.size > 0:
            cv2.imshow('Zoomed Detection', zoomed_frame)

            # บันทึกเฟรมขยายทุกๆ 10 วินาที
            current_time = time.time()
            if current_time - last_save_time >= 10:
                timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
                save_path = os.path.join(save_folder, f"zoomed_detection_{timestamp}.jpg")
                cv2.imwrite(save_path, zoomed_frame)
                last_save_time = current_time

                # ส่งข้อความแจ้งเตือน LINE Notify พร้อมรูปภาพ
                message = f"ตรวจพบการสูบบุหรี่เมื่อ {timestamp}"
                send_line_notify(message, save_path)

    # แสดงผลเฟรมที่มีการใส่กรอบและป้าย
    cv2.imshow('Webcam Object Detection', annotated_frame)

    # กด 'q' เพื่อออกจากการดูเว็บแคม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดการใช้งานเว็บแคมและหน้าต่างทั้งหมด
cap.release()
cv2.destroyAllWindows()
