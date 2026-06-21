import os
import pickle
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List
import numpy as np
import cv2
from .paths import groundingdino_python, project_root
from .perception import Detection

class SubprocessDetector:

    def __init__(self, gpu_id: int=0, groundingdino_env: str | None=None, max_retries: int=2, detection_timeout: int=1200):
        self.gpu_id = gpu_id
        self.groundingdino_python = str(groundingdino_env) if groundingdino_env else str(groundingdino_python())
        self.script_path = project_root() / 'scripts' / 'run_detection_groundingdino.py'
        self.max_retries = max_retries
        self.detection_timeout = detection_timeout

    def detect(self, images: List[np.ndarray], target_text: str, visualization_dir: str=None) -> List[Detection]:
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._detect_once(images, target_text, visualization_dir)
            except subprocess.CalledProcessError as e:
                last_error = e
                stderr = (e.stderr or b'').decode('utf-8', errors='replace')
                is_oom = 'out of memory' in stderr.lower() or 'OOM' in stderr or e.returncode in (137, 143)
                if attempt < self.max_retries and is_oom:
                    print(f'[SubprocessDetector] OOM/subprocess failure (attempt {attempt + 1}/{self.max_retries + 1}), retrying in 5s... stderr: {stderr[:200]}')
                    time.sleep(5)
                else:
                    raise
        raise last_error

    def _detect_once(self, images: List[np.ndarray], target_text: str, visualization_dir: str=None) -> List[Detection]:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for i, img in enumerate(images):
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(tmpdir / f'view_{i:02d}.jpg'), img_bgr)
            out_pkl = tmpdir / 'detections.pkl'
            python_exe = Path(self.groundingdino_python)
            cmd = [str(python_exe), str(self.script_path), '--images_dir', str(tmpdir), '--target_text', target_text, '--output_pickle', str(out_pkl), '--gpu_id', str(self.gpu_id)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.detection_timeout)
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
            with open(out_pkl, 'rb') as f:
                raw = pickle.load(f)
        detections = []
        for d in raw:
            detections.append(Detection(view_id=d['view_id'], bbox=tuple(d['bbox']), confidence=d['confidence'], mask=None))
        return detections