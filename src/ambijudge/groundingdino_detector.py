"""
GroundingDINO Local Detector
============================

Local GroundingDINO detector to replace DINO-X API calls.
"""
from __future__ import annotations
import sys
import os
import cv2
import numpy as np
import torch
from typing import List, Optional, Tuple
from dataclasses import dataclass
import supervision as sv
from .paths import ensure_grounding_dino_on_path, grounding_dino_config_path, grounding_dino_root, grounding_dino_weights_path
GROUNDING_DINO_ROOT = str(ensure_grounding_dino_on_path())
try:
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    GROUNDING_DINO_AVAILABLE = True
except ImportError as e:
    GROUNDING_DINO_AVAILABLE = False
    print(f'Warning: GroundingDINO not available: {e}')
    print('Please check GroundingDINO installation and C++ extensions.')
from .perception import Detection

@dataclass
class GroundingDINOConfig:
    """GroundingDINO configuration"""
    config_path: str = str(grounding_dino_config_path())
    weights_path: str = str(grounding_dino_weights_path())
    box_threshold: float = 0.5
    text_threshold: float = 0.5
    cpu_only: bool = False
    grounding_dino_root: str = str(grounding_dino_root())

class GroundingDINODetector:
    """
    Local GroundingDINO detector for object detection.
    Replaces DINO-X API calls with local inference.
    """

    def __init__(self, config: Optional[GroundingDINOConfig]=None):
        """
        Initialize GroundingDINO detector
        
        Args:
            config: GroundingDINO configuration
        """
        self.config = config or GroundingDINOConfig()
        self.grounding_dino_root = self.config.grounding_dino_root
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load GroundingDINO model"""
        if not GROUNDING_DINO_AVAILABLE:
            raise RuntimeError('GroundingDINO not available. Please install and configure it.')
        try:
            print(f'Loading GroundingDINO model from {self.config.weights_path}')
            self.model = load_model(self.config.config_path, self.config.weights_path, device='cuda' if not self.config.cpu_only else 'cpu')
            if not self.config.cpu_only and torch.cuda.is_available():
                self.model = self.model.cuda()
                print(f'GroundingDINO model moved to GPU: {next(self.model.parameters()).device}')
            else:
                print(f'GroundingDINO model on CPU: {next(self.model.parameters()).device}')
            print('GroundingDINO model loaded successfully')
        except Exception as e:
            raise RuntimeError(f'Failed to load GroundingDINO model: {e}')

    def detect_objects(self, image: np.ndarray, text_prompt: str, output_dir: str=None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Detect objects in image using GroundingDINO official demo script
        
        Args:
            image: Input image as numpy array (RGB format)
            text_prompt: Text description of objects to detect
            output_dir: Directory to save visualization results (optional)
        
        Returns:
            tuple: (boxes, logits, phrases)
                - boxes: numpy array of bounding boxes [N, 4] (x1, y1, x2, y2)
                - logits: numpy array of confidence scores [N]
                - phrases: list of detected object phrases [N]
        """
        try:
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            import time
            temp_image_path = f'/tmp/groundingdino_input_{int(time.time() * 1000)}.jpg'
            cv2.imwrite(temp_image_path, image_bgr)
            if output_dir is None:
                import tempfile
                temp_output = tempfile.mkdtemp(prefix='groundingdino_')
                cleanup_output = True
            else:
                os.makedirs(output_dir, exist_ok=True)
                temp_output = output_dir
                cleanup_output = False
            try:
                import subprocess
                cmd = ['python', os.path.join(self.grounding_dino_root, 'demo', 'inference_on_a_image.py'), '-c', self.config.config_path, '-p', self.config.weights_path, '-i', temp_image_path, '-o', temp_output, '-t', text_prompt, '--box_threshold', str(self.config.box_threshold), '--text_threshold', str(self.config.text_threshold)]
                env = os.environ.copy()
                env['PYTHONPATH'] = f'{self.grounding_dino_root}:{env.get('PYTHONPATH', '')}'
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.grounding_dino_root, env=env, timeout=120)
                if result.returncode != 0:
                    print(f'GroundingDINO demo failed: {result.stderr}')
                    return self._fallback_detection(image, text_prompt)
                boxes, logits, phrases = self._parse_demo_output(temp_output, text_prompt, image)
                if output_dir and len(boxes) > 0:
                    os.makedirs(output_dir, exist_ok=True)
                    original_image = cv2.imread(temp_image_path)
                    if original_image is not None:
                        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                        output_path = os.path.join(output_dir, 'annotated.jpg')
                        annotated_image = self.save_annotated_image(image_rgb, boxes, logits, phrases, output_path)
                        print(f'Visualization saved to: {output_dir}')
                return (boxes, logits, phrases)
            finally:
                if cleanup_output and os.path.exists(temp_output):
                    import shutil
                    shutil.rmtree(temp_output)
        except Exception as e:
            print(f'Warning: GroundingDINO detection failed: {e}')
            return self._fallback_detection(image, text_prompt)
        finally:
            if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
                os.remove(temp_image_path)

    def _parse_demo_output(self, output_dir: str, text_prompt: str, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Directly call GroundingDINO model instead of parsing demo output files
        
        Args:
            output_dir: Directory containing demo output files (not used in direct mode)
            text_prompt: Original text prompt
            image: Input image as numpy array
        
        Returns:
            tuple: (boxes, logits, phrases)
        """
        try:
            import sys
            sys.path.insert(0, self.grounding_dino_root)
            import torch
            import groundingdino.datasets.transforms as T
            from groundingdino.models import build_model
            from groundingdino.util.slconfig import SLConfig
            from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
            from PIL import Image
            if not hasattr(self, 'model') or self.model is None:
                print('Loading GroundingDINO model...')
                args = SLConfig.fromfile(self.config.config_path)
                args.device = 'cuda' if torch.cuda.is_available() and (not self.config.cpu_only) else 'cpu'
                self.model = build_model(args)
                checkpoint = torch.load(self.config.weights_path, map_location='cpu')
                load_res = self.model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
                print(f'Model loaded: {load_res}')
                self.model.eval()
            image_pil = Image.fromarray(image.astype(np.uint8))
            transform = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            image_tensor, _ = transform(image_pil, None)
            text_prompt = text_prompt.lower().strip()
            if not text_prompt.endswith('.'):
                text_prompt = text_prompt + '.'
            device = 'cuda' if torch.cuda.is_available() and (not self.config.cpu_only) else 'cpu'
            self.model = self.model.to(device)
            image_tensor = image_tensor.to(device)
            with torch.no_grad():
                outputs = self.model(image_tensor[None], captions=[text_prompt])
            logits = outputs['pred_logits'].sigmoid()[0]
            boxes = outputs['pred_boxes'][0]
            logits_filt = logits.cpu().clone()
            boxes_filt = boxes.cpu().clone()
            filt_mask = logits_filt.max(dim=1)[0] > self.config.box_threshold
            logits_filt = logits_filt[filt_mask]
            boxes_filt = boxes_filt[filt_mask]
            if len(logits_filt) == 0:
                return (np.array([]).reshape(0, 4), np.array([]), [])
            tokenizer = self.model.tokenizer
            tokenized = tokenizer(text_prompt)
            pred_phrases = []
            for logit, box in zip(logits_filt, boxes_filt):
                pred_phrase = get_phrases_from_posmap(logit > self.config.text_threshold, tokenized, tokenizer)
                pred_phrases.append(pred_phrase)
            H, W = (image_pil.size[1], image_pil.size[0])
            boxes_pixels = boxes_filt * torch.Tensor([W, H, W, H])
            boxes_xyxy = boxes_pixels.clone()
            boxes_xyxy[:, :2] -= boxes_xyxy[:, 2:] / 2
            boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]
            boxes_np = boxes_xyxy.numpy().astype(int)
            logits_np = logits_filt.max(dim=1)[0].numpy()
            print(f'GroundingDINO detected {len(boxes_np)} objects')
            return (boxes_np, logits_np, pred_phrases)
        except Exception as e:
            print(f'Warning: Direct GroundingDINO detection failed: {e}')
            import traceback
            traceback.print_exc()
            return (np.array([]).reshape(0, 4), np.array([]), [])

    def _fallback_detection(self, image: np.ndarray, text_prompt: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Fallback detection when GroundingDINO C++ extensions are not available
        
        Args:
            image: Input image
            text_prompt: Text prompt
        
        Returns:
            tuple: (boxes, logits, phrases) - empty results
        """
        print(f'Using fallback detection for prompt: {text_prompt}')
        boxes = np.array([]).reshape(0, 4)
        logits = np.array([])
        phrases = []
        return (boxes, logits, phrases)

    def detect(self, images: List[np.ndarray], target_text: str, visualization_dir: str=None) -> List[Detection]:
        """
        Detect objects in multiple images using optimized batch processing
        
        Args:
            images: List of images as numpy arrays
            target_text: Text description of objects to detect
            visualization_dir: Base directory to save visualization results
        
        Returns:
            List of Detection objects
        """
        print(f'GroundingDINO batch detection starting:')
        print(f"  - Text prompt: '{target_text}'")
        print(f'  - Number of images: {len(images)}')
        print(f'  - Visualization directory: {visualization_dir}')
        return self._detect_batch_optimized(images, target_text, visualization_dir)

    def _detect_batch_optimized(self, images: List[np.ndarray], target_text: str, visualization_dir: str=None) -> List[Detection]:
        """
        Optimized batch detection using GroundingDINO model directly
        """
        import torch
        import torchvision.transforms as T
        from PIL import Image
        import time
        all_detections = []
        print('Preprocessing images for batch detection...')
        start_time = time.time()
        target_size = 800
        transform = T.Compose([T.Resize((target_size, target_size)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        processed_images = []
        original_shapes = []
        for i, image in enumerate(images):
            if len(image.shape) == 3 and image.shape[2] == 3:
                pil_image = Image.fromarray(image.astype(np.uint8))
            else:
                pil_image = Image.fromarray(image.astype(np.uint8))
            original_shapes.append(image.shape[:2])
            tensor_image = transform(pil_image)
            processed_images.append(tensor_image)
        if len(processed_images) == 0:
            print('Warning: No images to process')
            return []
        print(f'Debug: Processing {len(processed_images)} images')
        for i, img in enumerate(processed_images[:3]):
            print(f'  Image {i}: shape={img.shape}, dtype={img.dtype}')
        try:
            batch_tensor = torch.stack(processed_images)
            print(f'Debug: Batch tensor shape: {batch_tensor.shape}')
        except Exception as e:
            print(f'Debug: Stack failed: {e}')
            print(f'Debug: First image shape: {(processed_images[0].shape if processed_images else 'None')}')
            print(f'Debug: All shapes: {[img.shape for img in processed_images[:5]]}')
            raise e
        if torch.cuda.is_available() and (not self.config.cpu_only):
            batch_tensor = batch_tensor.cuda()
        preprocess_time = time.time() - start_time
        print(f'Image preprocessing completed in {preprocess_time:.3f}s')
        print('Running batch inference...')
        inference_start = time.time()
        try:
            with torch.no_grad():
                text_prompt = target_text.lower().strip()
                from groundingdino.util.misc import nested_tensor_from_tensor_list
                tensor_list = [batch_tensor[i] for i in range(batch_tensor.shape[0])]
                nested_tensor = nested_tensor_from_tensor_list(tensor_list)
                outputs = self.model(nested_tensor, captions=[text_prompt] * len(images))
                for view_id in range(len(images)):
                    boxes = outputs['pred_boxes'][view_id].cpu().numpy()
                    logits = outputs['pred_logits'][view_id].cpu().numpy()
                    confidence_threshold = 0.3
                    valid_indices = logits > confidence_threshold
                    if np.any(valid_indices):
                        valid_boxes = boxes[valid_indices]
                        valid_logits = logits[valid_indices]
                        phrases = [text_prompt] * len(valid_boxes)
                        print(f'  - View {view_id}: Found {len(valid_boxes)} detections')
                        print(f'    Confidence range: {valid_logits.min():.4f} - {valid_logits.max():.4f}')
                        orig_h, orig_w = original_shapes[view_id]
                        scale_x = orig_w / target_size
                        scale_y = orig_h / target_size
                        for box, logit in zip(valid_boxes, valid_logits):
                            x1, y1, x2, y2 = box
                            x1 = int(x1 * orig_w)
                            y1 = int(y1 * orig_h)
                            x2 = int(x2 * orig_w)
                            y2 = int(y2 * orig_h)
                            x, y, w, h = (x1, y1, x2 - x1, y2 - y1)
                            detection = Detection(view_id=view_id, bbox=(x, y, w, h), confidence=float(logit), mask=None)
                            all_detections.append(detection)
                    else:
                        print(f'  - View {view_id}: No detections above threshold')
        except Exception as e:
            print(f'Batch inference failed, falling back to sequential processing: {e}')
            return self._detect_sequential_fallback(images, target_text, visualization_dir)
        inference_time = time.time() - inference_start
        total_time = time.time() - start_time
        print(f'Batch inference completed in {inference_time:.3f}s')
        print(f'Total detection time: {total_time:.3f}s')
        print(f'Average per image: {total_time / len(images):.3f}s')
        print(f'GroundingDINO detection completed: {len(all_detections)} total detections')
        return all_detections

    def _detect_sequential_fallback(self, images: List[np.ndarray], target_text: str, visualization_dir: str=None) -> List[Detection]:
        """
        Optimized sequential processing with faster image preprocessing
        """
        print('Using optimized sequential processing...')
        all_detections = []
        print('Pre-resizing images for faster processing...')
        resized_images = []
        original_shapes = []
        for image in images:
            h, w = image.shape[:2]
            max_size = 512
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = (int(h * scale), int(w * scale))
                resized_image = cv2.resize(image, (new_w, new_h))
            else:
                resized_image = image
                scale = 1.0
            resized_images.append(resized_image)
            original_shapes.append((h, w, scale))
        print(f'Image resizing completed, processing {len(resized_images)} images...')
        for view_id, (resized_image, (orig_h, orig_w, scale)) in enumerate(zip(resized_images, original_shapes)):
            if view_id % 5 == 0:
                print(f'Processing view {view_id}/{len(images)} (image shape: {resized_image.shape})')
            try:
                view_output_dir = None
                boxes, logits, phrases = self.detect_objects(resized_image, target_text, view_output_dir)
                if len(boxes) > 0:
                    if view_id % 5 == 0:
                        print(f'  - Found {len(boxes)} detections in view {view_id}')
                        print(f'  - Confidence range: {min(logits):.4f} - {max(logits):.4f}')
                        print(f'  - Phrases: {phrases}')
                for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
                    x1, y1, x2, y2 = box
                    x1 = int(x1 / scale)
                    y1 = int(y1 / scale)
                    x2 = int(x2 / scale)
                    y2 = int(y2 / scale)
                    x, y, w, h = (x1, y1, x2 - x1, y2 - y1)
                    detection = Detection(view_id=view_id, bbox=(x, y, w, h), confidence=float(logit), mask=None)
                    all_detections.append(detection)
            except Exception as e:
                print(f'Warning: Detection failed for view {view_id}: {e}')
        print(f'Optimized sequential detection completed: {len(all_detections)} total detections')
        return all_detections

    def save_annotated_image(self, image: np.ndarray, boxes: np.ndarray, logits: np.ndarray, phrases: List[str], output_path: str) -> np.ndarray:
        """
        Save annotated image with detection results
        
        Args:
            image: Original image
            boxes: Detection boxes
            logits: Confidence scores
            phrases: Object phrases
            output_path: Output image path
        
        Returns:
            Annotated image as numpy array
        """
        try:
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            temp_path = '/tmp/temp_groundingdino_annotate.jpg'
            cv2.imwrite(temp_path, image_bgr)
            image_source, _ = load_image(temp_path)
            if len(boxes) > 0:
                H, W = image.shape[:2]
                boxes_normalized = boxes.copy().astype(float)
                boxes_normalized[:, [0, 2]] /= W
                boxes_normalized[:, [1, 3]] /= H
                annotated_frame = annotate(image_source=image_source, boxes=boxes_normalized, logits=logits, phrases=phrases)
            else:
                annotated_frame = image_source
            cv2.imwrite(output_path, annotated_frame)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return annotated_frame
        except Exception as e:
            print(f'Warning: Failed to save annotated image: {e}')
            return image

def create_groundingdino_detector(config: Optional[GroundingDINOConfig]=None) -> GroundingDINODetector:
    """
    Factory function to create GroundingDINO detector
    
    Args:
        config: Optional configuration
    
    Returns:
        GroundingDINODetector instance
    """
    return GroundingDINODetector(config)
DINOXDetector = GroundingDINODetector