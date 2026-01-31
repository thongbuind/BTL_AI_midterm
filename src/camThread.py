import cv2
import subprocess
import threading
import numpy as np
import time
from visual import FPSCalculator

class CamThread:
    def __init__(
        self,
        cam_name,
        source,
        mode="rtsp",
        target_size=(640, 480),
        target_fps=20
    ):
        self.cam_name = cam_name
        self.source = source
        self.mode = mode
        self.target_size = target_size
        self.target_fps = target_fps

        self.frame = None
        self.running = True
        self.lock = threading.Lock()

        self.fps_calc = FPSCalculator()
        self.last_capture_time = 0
        self.frame_interval = 1.0 / target_fps

        if self.mode == "webcam":
            self._init_webcam()
        elif self.mode == "rtsp":
            self._init_rtsp_hw()
        else:
            raise ValueError(f"Unknown camera mode: {self.mode}")

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    # =========================
    # INIT
    # =========================

    def _init_webcam(self):
        print(f"[{self.cam_name}] Webcam - Target FPS: {self.target_fps}")
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_size[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

    def _init_rtsp_hw(self):
        print(f"[{self.cam_name}] RTSP + VideoToolbox - Target FPS: {self.target_fps}")

        w, h = self.target_size
        self.frame_size = w * h * 3

        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-hwaccel", "videotoolbox",
            "-rtsp_transport", "tcp",
            "-i", self.source,
            "-vf", f"scale={w}:{h}",
            "-r", str(self.target_fps),
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "-"
        ]

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self.frame_size * 2
        )

    # =========================
    # UPDATE LOOP
    # =========================

    def update(self):
        while self.running:
            current_time = time.time()
            if current_time - self.last_capture_time < self.frame_interval:
                time.sleep(0.001)
                continue

            if self.mode == "webcam":
                ret, frame = self.cap.read()
                if not ret:
                    continue
                frame = cv2.resize(frame, self.target_size)

            elif self.mode == "rtsp":
                raw = self.proc.stdout.read(self.frame_size)
                if not raw or len(raw) < self.frame_size:
                    continue

                frame = np.frombuffer(
                    raw[:self.frame_size], dtype=np.uint8
                ).reshape(
                    (self.target_size[1], self.target_size[0], 3)
                )

            else:
                continue

            with self.lock:
                self.frame = frame
                self.last_capture_time = current_time
                self.fps_calc.update()

    # =========================
    # PUBLIC API
    # =========================

    def read(self):
        with self.lock:
            if self.frame is None:
                return None, 0.0
            return self.frame.copy(), self.fps_calc.get_fps()

    def release(self):
        self.running = False

        if hasattr(self, "cap"):
            self.cap.release()

        if hasattr(self, "proc"):
            try:
                self.proc.kill()
            except Exception:
                pass

        print(f"[{self.cam_name}] Released.")
