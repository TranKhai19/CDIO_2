from ultralytics import YOLO
import cv2
import face_recognition
import os
import csv
from datetime import datetime

# Tải mô hình YOLOv8
model = YOLO("yolov8n.pt")

known_encodings = []
known_names = []

# Hàm để phát hiện khuôn mặt sử dụng YOLOv8
def detect_faces(image):
    results = model(image)
    boxes = results[0].boxes.xyxy  # Lấy toạ độ khung hình các khuôn mặt
    return boxes


# Hàm nhận diện khuôn mặt
def recognize_faces(known_encodings, known_names, frame):
    # Phát hiện các vị trí khuôn mặt trong khung hình
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Duyệt qua từng khuôn mặt được phát hiện
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # So sánh khuôn mặt được mã hóa với các khuôn mặt đã biết
        matches = face_recognition.compare_faces(known_encodings, face_encoding)

        name = "Unknown"  # Mặc định là "Unknown" nếu không tìm thấy kết quả

        top = int(top)
        right = int(right)
        bottom = int(bottom)
        left = int(left)

        # Nếu tìm thấy kết quả phù hợp
        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

        # Vẽ hình chữ nhật xung quanh khuôn mặt
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Hiển thị tên dưới khuôn mặt
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return frame


employee_dir = "employees_photos"
for filename in os.listdir(employee_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Kiểm tra định dạng ảnh
        # Lấy tên từ file (không bao gồm phần mở rộng)
        name = os.path.splitext(filename)[0]

        # Đọc ảnh
        image_path = os.path.join(employee_dir, filename)
        image = face_recognition.load_image_file(image_path)

        # Mã hóa khuôn mặt
        face_encoding = face_recognition.face_encodings(image)

        if face_encoding:
            # Thêm mã hóa và tên vào danh sách
            known_encodings.append(face_encoding[0])
            known_names.append(name)

print(f"Đã mã hóa {len(known_encodings)} khuôn mặt.")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = recognize_faces(known_encodings, known_names, frame)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Hàm ghi kết quả điểm danh
def save_attendance(name):
    with open('attendance.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])