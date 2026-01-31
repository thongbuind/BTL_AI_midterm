import cv2
import numpy as np
import time
from collections import deque

class FPSCalculator:
    def __init__(self, window_size=30):
        self.timestamps = deque(maxlen=window_size)
    
    def update(self):
        self.timestamps.append(time.time())
    
    def get_fps(self):
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return len(self.timestamps) / elapsed if elapsed > 0 else 0.0


def make_grid(frames, grid_size=(640, 480)):
    grid_frames = []
    
    for i in range(4):
        cam_name = f"cam{i}"
        if cam_name in frames:
            frame = frames[cam_name]
            frame = cv2.resize(frame, grid_size)
            grid_frames.append(frame)
        else:
            blank = np.zeros((grid_size[1], grid_size[0], 3), dtype=np.uint8)
            cv2.putText(
                blank,
                f"CAM {i} - NO SIGNAL",
                (50, grid_size[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            grid_frames.append(blank)
    
    top = np.hstack((grid_frames[0], grid_frames[1]))
    bottom = np.hstack((grid_frames[2], grid_frames[3]))
    grid = np.vstack((top, bottom))
    
    return grid
