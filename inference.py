import cv2
from ultralytics import YOLO

# 定义模型
low_energy_model = YOLO(r'C:\Users\ASUS\Desktop\Eco-Sight-main\runs\detect\yolov10n5\weights\best.pt')
medium_energy_model = YOLO(r'C:\Users\ASUS\Desktop\Eco-Sight-main\runs\detect\yolov10s6\weights\best.pt')
high_energy_model = YOLO(r'C:\Users\ASUS\Desktop\Eco-Sight-main\runs\detect\yolov10m5\weights\best.pt')

# 给模型对象添加一个name属性
low_energy_model.name = 'Low Energy'
medium_energy_model.name = 'Medium Energy'
high_energy_model.name = 'High Energy'

# 初始化当前模型为中等能耗模型
current_model = medium_energy_model
frame_counter = 0
frame_rate = 5
max_frame_rate = 30

# 打开视频文件
cap = cv2.VideoCapture(r"C:\Users\ASUS\Desktop\WeChat_20241016193021.mp4")

# 人和车的class_id，根据你的names列表来设置
person_class_id = 7  # 'human hauler'
vehicle_class_ids = [4, 5, 6, 15, 16, 18]  # 'bus', 'car', 'minibus', 'pickup', 'suv', 'truck'

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_counter % frame_rate == 0:
        results = current_model(frame)
        detections = results[0].boxes
        detected_person_or_vehicle = False
        detected_objects_count = 0

        for detection in detections:
            class_id = int(detection.cls)
            confidence = detection.conf

            # 检查是否检测到人或车
            if confidence > 0.5 and (class_id == person_class_id or class_id in vehicle_class_ids):
                detected_person_or_vehicle = True
                detected_objects_count += 1

        # 根据检测结果切换模型
        if not detected_person_or_vehicle:
            current_model = low_energy_model
            frame_rate = 5
            print(f"No person or vehicle detected. Switched to {current_model.name} model.")
        elif detected_objects_count <= 3:
            current_model = high_energy_model
            frame_rate = max_frame_rate
            print(f"Detected {detected_objects_count} objects. Switched to {current_model.name} model.")
        else:
            current_model = medium_energy_model
            frame_rate = 5
            print(f"Detected {detected_objects_count} objects. Switched to {current_model.name} model.")

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_counter += 1

cap.release()
cv2.destroyAllWindows()