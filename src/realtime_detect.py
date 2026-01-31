import cv2
import json
import time
from pathlib import Path
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from camThread import CamThread
from visual import make_grid

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
config_file = project_root / "config.json"
model_file = project_root / "model" / "best.pt"

with config_file.open("r") as f:
    cfg = json.load(f)

image_size = cfg.get("image_size", 640)
classes = cfg.get("classes", {})

model = YOLO(model_file)
model.to("mps:0")
model.fuse()

RTSP_URLS = {
    "cam0": "rtsp://admin:TIJEQB@192.168.1.134:554/ch1/main",
    "cam1": "rtsp://admin:TIJEQB@192.168.1.134:554/ch1/main",
    "cam2": "rtsp://admin:TIJEQB@192.168.1.134:554/ch1/main",
    "cam3": "rtsp://admin:TIJEQB@192.168.1.134:554/ch1/main",
}

print(f"\n📹 [CAMERAS] Initializing {len(RTSP_URLS)} cameras...")
cam_threads = {}
for name, url in RTSP_URLS.items():
    cam_threads[name] = CamThread(
        cam_name=name,
        source=url,
        mode="rtsp",
        target_size=(640, 640),
        target_fps=15 
    )
executor = ThreadPoolExecutor(max_workers=4)

def process_single_cam(cam_name, frame, result):
    annotated = frame.copy()
    
    if result.boxes is None or len(result.boxes) == 0:
        # Không có detection
        cv2.putText(
            annotated,
            cam_name.upper(),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )
        return annotated
    
    boxes = result.boxes
    
    # Vẽ từng box
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes.xyxy[i])
        conf = float(boxes.conf[i])
        cls = int(boxes.cls[i])
        
        # Lấy class name
        class_name = model.names.get(cls, f"class_{cls}")
        label = f"{class_name} {conf:.2f}"
        
        # Vẽ bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    
    # Thêm cam label
    cv2.putText(
        annotated,
        f"{cam_name.upper()} | {len(boxes)} objs",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )
    
    return annotated

def process_batch_results(frames, batch_results, cam_names):
    futures = []
    
    for cam_name, result in zip(cam_names, batch_results):
        frame = frames[cam_name]
        future = executor.submit(process_single_cam, cam_name, frame, result)
        futures.append((cam_name, future))
    
    processed_frames = {}
    for cam_name, future in futures:
        processed_frames[cam_name] = future.result()
    
    return processed_frames

frame_count = 0
start_time = time.time()

try:
    while True:
        frames = {}
        for name, cam in cam_threads.items():
            frame, fps = cam.read()
            if frame is not None:
                frames[name] = frame
        
        if len(frames) != 4:
            continue
        
        cam_names = list(frames.keys())
        frame_list = [frames[n] for n in cam_names]
        
        batch_results = model(
            frame_list,
            imgsz=image_size,
            conf=0.5,
            device="mps:0",
            verbose=False
        )
        
        processed_frames = process_batch_results(frames, batch_results, cam_names)
        
        grid = make_grid(processed_frames)
        
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"📊 Frame {frame_count} | FPS: {fps:.2f}")
        
        cv2.imshow("4-CAM Detection", grid)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n Stopped by user")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("\n[INFO] Shutting down...")
    for cam in cam_threads.values():
        cam.release()
    executor.shutdown(wait=True)
    cv2.destroyAllWindows()
    
    if frame_count > 0:
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        print(f"✅ Processed {frame_count} frames in {total_time:.1f}s ({avg_fps:.2f} FPS)")
