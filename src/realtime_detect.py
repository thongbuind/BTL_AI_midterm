import cv2
import json
from pathlib import Path
from ultralytics import YOLO

import cv2
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
model_file = project_root / "model" / "best.py"
model = YOLO(model_file = project_root / "model" / "best.py")

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
config_file = project_root / "config.json"
src_dir = project_root / "src"

url = "rtsp://admin:PBPBND@192.168.0.119:554/ch1/main"

cam = cv2.VideoCapture(url)

if not cam.isOpened():
    print("❌ Không mở được camera!")
    exit()

with open(config_file, 'r') as f:
    config = json.load(f)
image_size = config["image_size"]

while True:
    ret, frame = cam.read()
    if not ret:
        break

    results = model(frame, stream=True, imgsz=image_size, conf=0.7)

    detected_classes = set()
    for r in results:
        boxes = r.boxes
        for cls in boxes.cls:
            class_name = model.names[int(cls)]
            detected_classes.add(class_name)
        annotated_frame = r.plot()

    cv2.imshow("YOLOv8 Real-time Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
