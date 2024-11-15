import cv2
import keyboard
import os
import time

# ตั้งค่าโฟลเดอร์ที่ต้องการบันทึกภาพ
save_folder = 'D:\\petch\\cap'

# ตรวจสอบว่าโฟลเดอร์มีอยู่หรือไม่ ถ้าไม่มีก็สร้างใหม่
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# เปิดกล้อง webcam
cap = cv2.VideoCapture(0)

# ตัวแปรสำหรับนับจำนวนภาพ
image_count = 0

# ตัวแปรสำหรับตรวจสอบการกดปุ่ม
key_pressed = False

while True:
    # อ่านภาพจากกล้อง
    ret, frame = cap.read()

    if not ret:
        print("ไม่สามารถจับภาพจากกล้องได้")
        break

    # แสดงภาพจากกล้อง
    cv2.imshow('Webcam', frame)

    # ตรวจสอบการกดปุ่ม 'f'
    if keyboard.is_pressed('f'):
        if not key_pressed:
            # สร้างชื่อไฟล์ที่มี timestamp
            filename = os.path.join(save_folder, 'capture_{}.jpg'.format(image_count))
            # บันทึกภาพลงในไฟล์
            cv2.imwrite(filename, frame)
            image_count += 1
            print(f'บันทึกภาพลงใน {filename}')
            print(f'จำนวนภาพที่บันทึก: {image_count}')
            key_pressed = True
            time.sleep(0.1)  # Delay to avoid multiple captures

    else:
        key_pressed = False

    # ออกจากลูปเมื่อกดปุ่ม 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปล่อยกล้องและปิดหน้าต่าง
cap.release()
cv2.destroyAllWindows()
