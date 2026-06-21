import torch
import numpy as np
import cv2
import time
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import torchvision.transforms as T
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.inference import preprocess_caption, load_model
from groundingdino.datasets.transforms import resize
from .paths import grounding_dino_config_path, grounding_dino_weights_path
from .perception import Detection

class Resize(object):

    def __init__(self, size):
        assert isinstance(size, (list, tuple))
        self.size = size

    def __call__(self, img, target=None):
        rescaled_image, rescaled_target = resize(img, target, self.size)
        return (rescaled_image, rescaled_target)

class BatchGroundingDINODetector:

    def __init__(self, config_path: str=str(grounding_dino_config_path()), weights_path: str=str(grounding_dino_weights_path()), device: str='cuda', batch_size: int=8, box_threshold: float=0.6, text_threshold: float=0.25):
        self.config_path = config_path
        self.weights_path = weights_path
        self.device = device
        self.batch_size = batch_size
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        import os
        original_proxy = {}
        for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
            if key in os.environ:
                original_proxy[key] = os.environ[key]
                del os.environ[key]
        os.environ['TRANSFORMERS_OFFLINE'] = '0'
        os.environ['HF_HUB_OFFLINE'] = '0'
        print('Loading GroundingDINO model for batch processing...')
        try:
            self.model = load_model(config_path, weights_path, device=device)
        finally:
            for key, value in original_proxy.items():
                os.environ[key] = value
        self.model.eval()
        print(f'GroundingDINO model loaded on {device}')
        self.resize_transform = Resize((800, 1200))
        self.tensor_transform = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.last_timing_info = None

    def load_image_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        processed_images = []
        for img in images:
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            resized_img, _ = self.resize_transform(pil_img, None)
            img_tensor = self.tensor_transform(resized_img)
            processed_images.append(img_tensor)
        return torch.stack(processed_images)

    def predict_batch(self, images: torch.Tensor, caption: str) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]]]:
        caption = preprocess_caption(caption=caption)
        images = images.to(self.device)
        self.model = self.model.to(self.device)
        with torch.no_grad():
            outputs = self.model(images, captions=[caption for _ in range(len(images))])
        prediction_logits = outputs['pred_logits'].cpu().sigmoid()
        prediction_boxes = outputs['pred_boxes'].cpu()
        bboxes_batch = []
        logits_batch = []
        phrases_batch = []
        tokenizer = self.model.tokenizer
        tokenized = tokenizer(caption)
        for i in range(prediction_logits.shape[0]):
            mask = prediction_logits[i].max(dim=1)[0] > self.box_threshold
            if mask.sum() == 0:
                bboxes_batch.append(torch.empty(0, 4))
                logits_batch.append(torch.empty(0))
                phrases_batch.append([])
                continue
            logits = prediction_logits[i][mask]
            boxes = prediction_boxes[i][mask]
            phrases = [get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenizer).replace('.', '') for logit in logits]
            bboxes_batch.append(boxes)
            logits_batch.append(logits.max(dim=1)[0])
            phrases_batch.append(phrases)
        return (bboxes_batch, logits_batch, phrases_batch)

    def convert_to_detections(self, bboxes_batch: List[torch.Tensor], logits_batch: List[torch.Tensor], phrases_batch: List[List[str]], original_images: List[np.ndarray], start_view_id: int=0) -> List[Detection]:
        detections = []
        for i, (bboxes, logits, phrases, img) in enumerate(zip(bboxes_batch, logits_batch, phrases_batch, original_images)):
            view_id = start_view_id + i
            h, w = img.shape[:2]
            for bbox, logit, phrase in zip(bboxes, logits, phrases):
                cx, cy, bw, bh = bbox
                x = cx - bw / 2
                y = cy - bh / 2
                x1 = int(x * w)
                y1 = int(y * h)
                x2 = int((x + bw) * w)
                y2 = int((y + bh) * h)
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w - 1, x2))
                y2 = max(0, min(h - 1, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                x, y, w, h = (x1, y1, x2 - x1, y2 - y1)
                detection = Detection(view_id=view_id, bbox=(x, y, w, h), confidence=float(logit), mask=None)
                detections.append(detection)
        return detections

    def detect(self, images: List[np.ndarray], target_text: str, visualization_dir: str=None) -> List[Detection]:
        print(f"Batch detection starting: {len(images)} images, target: '{target_text}'")
        print(f'Batch size: {self.batch_size}')
        all_detections = []
        batch_times = []
        total_start = time.time()
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            batch_start_id = i
            batch_start = time.time()
            print(f'Processing batch {i // self.batch_size + 1}/{(len(images) + self.batch_size - 1) // self.batch_size}: {len(batch_images)} images')
            start_time = time.time()
            batch_tensor = self.load_image_batch(batch_images)
            preprocess_time = time.time() - start_time
            print(f'  Preprocessing: {preprocess_time:.3f}s')
            start_time = time.time()
            bboxes_batch, logits_batch, phrases_batch = self.predict_batch(batch_tensor, target_text)
            inference_time = time.time() - start_time
            print(f'  Inference: {inference_time:.3f}s')
            start_time = time.time()
            batch_detections = self.convert_to_detections(bboxes_batch, logits_batch, phrases_batch, batch_images, batch_start_id)
            convert_time = time.time() - start_time
            print(f'  Convert: {convert_time:.3f}s')
            batch_total_time = preprocess_time + inference_time + convert_time
            batch_times.append(batch_total_time)
            print(f'  Total batch time: {batch_total_time:.3f}s')
            print(f'  Avg time per image in batch: {batch_total_time / len(batch_images):.3f}s')
            print(f'  Detections: {len(batch_detections)}')
            all_detections.extend(batch_detections)
        total_time = time.time() - total_start
        avg_time_per_image = total_time / len(images) if len(images) > 0 else 0.0
        print(f'Batch detection completed: {len(all_detections)} total detections')
        print(f'Total time: {total_time:.3f}s, Average per image: {avg_time_per_image:.3f}s')
        self.last_timing_info = {'total_time': total_time, 'avg_time_per_image': avg_time_per_image, 'num_images': len(images), 'num_batches': len(batch_times), 'batch_times': batch_times, 'avg_batch_time': sum(batch_times) / len(batch_times) if batch_times else 0.0}
        if visualization_dir is not None:
            try:
                import os
                os.makedirs(visualization_dir, exist_ok=True)
                for det in all_detections:
                    vid = det.view_id
                    if 0 <= vid < len(images):
                        img = images[vid].copy()
                        x, y, w, h = det.bbox
                        import cv2
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.imwrite(os.path.join(visualization_dir, f'view_{vid:02d}.jpg'), img_bgr)
                print(f'Saved quick visualizations to {visualization_dir}')
            except Exception as e:
                print(f'Visualization saving failed: {e}')
        return all_detections

def test_batch_detector():
    print('Testing batch GroundingDINO detector')
    print('=' * 50)

    def create_test_images(num_images: int=31) -> List[np.ndarray]:
        images = []
        for i in range(num_images):
            img = np.ones((480, 640, 3), dtype=np.uint8) * 255
            center_x, center_y = (320, 240)
            cv2.rectangle(img, (center_x - 60, center_y - 80), (center_x + 60, center_y - 20), (139, 69, 19), -1)
            cv2.rectangle(img, (center_x - 80, center_y - 20), (center_x + 80, center_y + 20), (139, 69, 19), -1)
            cv2.rectangle(img, (center_x - 70, center_y + 20), (center_x - 50, center_y + 120), (139, 69, 19), -1)
            cv2.rectangle(img, (center_x + 50, center_y + 20), (center_x + 70, center_y + 120), (139, 69, 19), -1)
            if i > 0:
                angle = i * 0.1
                M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
                img = cv2.warpAffine(img, M, (640, 480))
            images.append(img)
        return images
    detector = BatchGroundingDINODetector(batch_size=8)
    images = create_test_images(31)
    start_time = time.time()
    detections = detector.detect(images, 'chair')
    total_time = time.time() - start_time
    print('\n=== Performance ===')
    print(f'Total time: {total_time:.2f}s')
    print(f'Avg per frame: {total_time / len(images):.3f}s')
    print(f'Detections: {len(detections)}')
    print(f'Target met: {"yes" if total_time < 10 else "no"} (target: <10s)')
    return (detections, total_time)
if __name__ == '__main__':
    test_batch_detector()