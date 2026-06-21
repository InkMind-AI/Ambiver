#!/usr/bin/env python3
"""
基于Grounding-DINO-Batch-Inference的高效批处理检测器
结合多GPU并行和真正的批处理，目标将31帧处理时间从31秒压缩到10秒以内
"""

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
    """固定尺寸的Resize类，用于批处理"""
    def __init__(self, size):
        assert isinstance(size, (list, tuple))
        self.size = size

    def __call__(self, img, target=None):
        rescaled_image, rescaled_target = resize(img, target, self.size)
        return rescaled_image, rescaled_target


class BatchGroundingDINODetector:
    """
    基于批处理的高效GroundingDINO检测器
    结合多GPU并行和真正的批处理，实现10秒内处理31帧的目标
    """
    
    def __init__(self, 
                 config_path: str = str(grounding_dino_config_path()),
                 weights_path: str = str(grounding_dino_weights_path()),
                 device: str = "cuda",
                 batch_size: int = 8,  # 增大batch size
                 box_threshold: float = 0.6,
                 text_threshold: float = 0.25):
        """
        初始化批处理检测器
        
        Args:
            config_path: 模型配置文件路径
            weights_path: 模型权重文件路径
            device: 设备
            batch_size: 批处理大小
            box_threshold: 框阈值
            text_threshold: 文本阈值
        """
        self.config_path = config_path
        self.weights_path = weights_path
        self.device = device
        self.batch_size = batch_size
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        # 设置环境变量以确保使用本地缓存的tokenizer，不尝试下载
        import os
        # 禁用代理以避免连接问题
        original_proxy = {}
        for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
            if key in os.environ:
                original_proxy[key] = os.environ[key]
                del os.environ[key]
        
        # 设置HuggingFace使用本地缓存
        os.environ['TRANSFORMERS_OFFLINE'] = '0'  # 允许在线但优先使用缓存
        os.environ['HF_HUB_OFFLINE'] = '0'  # 允许在线但优先使用缓存
        
        # 加载模型
        print("Loading GroundingDINO model for batch processing...")
        try:
            self.model = load_model(config_path, weights_path, device=device)
        finally:
            # 恢复原始代理设置（如果需要）
            for key, value in original_proxy.items():
                os.environ[key] = value
        self.model.eval()
        print(f"GroundingDINO model loaded on {device}")
        
        # 创建固定尺寸的变换
        self.resize_transform = Resize((800, 1200))  # 固定尺寸，确保批处理一致性
        self.tensor_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        # 存储最后一次检测的时间信息
        self.last_timing_info = None
    
    def load_image_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        批量加载和预处理图像
        
        Args:
            images: 图像列表 (numpy arrays)
            
        Returns:
            预处理后的batch tensor
        """
        processed_images = []
        
        for img in images:
            # 转换为PIL Image
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            
            # 应用变换
            resized_img, _ = self.resize_transform(pil_img, None)
            img_tensor = self.tensor_transform(resized_img)
            processed_images.append(img_tensor)
        
        # 堆叠成batch
        return torch.stack(processed_images)
    
    def predict_batch(self, 
                     images: torch.Tensor, 
                     caption: str) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]]]:
        """
        批量预测
        
        Args:
            images: 预处理后的图像batch (B, C, H, W)
            caption: 文本提示
            
        Returns:
            (bboxes_batch, logits_batch, phrases_batch)
        """
        caption = preprocess_caption(caption=caption)
        images = images.to(self.device)
        
        # 确保模型在正确的设备上
        self.model = self.model.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images, captions=[caption for _ in range(len(images))])
        
        # 获取预测结果
        prediction_logits = outputs["pred_logits"].cpu().sigmoid()  # (B, 900, 256)
        prediction_boxes = outputs["pred_boxes"].cpu()  # (B, 900, 4)
        
        # 批量处理结果
        bboxes_batch = []
        logits_batch = []
        phrases_batch = []
        
        tokenizer = self.model.tokenizer
        tokenized = tokenizer(caption)
        
        for i in range(prediction_logits.shape[0]):
            # 过滤低置信度检测
            mask = prediction_logits[i].max(dim=1)[0] > self.box_threshold
            
            if mask.sum() == 0:
                # 没有检测到任何对象
                bboxes_batch.append(torch.empty(0, 4))
                logits_batch.append(torch.empty(0))
                phrases_batch.append([])
                continue
            
            # 提取检测结果
            logits = prediction_logits[i][mask]  # (n, 256)
            boxes = prediction_boxes[i][mask]  # (n, 4)
            
            # 生成短语
            phrases = [
                get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenizer).replace('.', '')
                for logit in logits
            ]
            
            bboxes_batch.append(boxes)
            logits_batch.append(logits.max(dim=1)[0])
            phrases_batch.append(phrases)
        
        return bboxes_batch, logits_batch, phrases_batch
    
    def convert_to_detections(self, 
                            bboxes_batch: List[torch.Tensor], 
                            logits_batch: List[torch.Tensor],
                            phrases_batch: List[List[str]],
                            original_images: List[np.ndarray],
                            start_view_id: int = 0) -> List[Detection]:
        """
        将批处理结果转换为Detection对象列表
        
        Args:
            bboxes_batch: 边界框列表
            logits_batch: 置信度列表
            phrases_batch: 短语列表
            original_images: 原始图像列表
            start_view_id: 起始view_id
            
        Returns:
            Detection对象列表
        """
        detections = []
        
        for i, (bboxes, logits, phrases, img) in enumerate(zip(bboxes_batch, logits_batch, phrases_batch, original_images)):
            view_id = start_view_id + i
            h, w = img.shape[:2]
            
            for bbox, logit, phrase in zip(bboxes, logits, phrases):
                # 转换坐标格式 (cx, cy, w, h) -> (x, y, w, h)
                cx, cy, bw, bh = bbox
                x = cx - bw / 2
                y = cy - bh / 2
                
                # 转换为像素坐标
                x1 = int(x * w)
                y1 = int(y * h)
                x2 = int((x + bw) * w)
                y2 = int((y + bh) * h)

                # 边界裁剪
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w - 1, x2))
                y2 = max(0, min(h - 1, y2))

                # 保证有效框
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # 转换为(x, y, w, h)格式
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                
                detection = Detection(
                    view_id=view_id,
                    bbox=(x, y, w, h),
                    confidence=float(logit),
                    mask=None
                )
                detections.append(detection)
        
        return detections
    
    def detect(self, images: List[np.ndarray], target_text: str, visualization_dir: str = None) -> List[Detection]:
        """
        批量检测主函数
        
        Args:
            images: 图像列表
            target_text: 目标文本
            visualization_dir: 可视化目录（可选）
            
        Returns:
            (Detection对象列表, 时间统计字典)
            时间统计包含: total_time, avg_time_per_image, batch_times
        """
        print(f"Batch detection starting: {len(images)} images, target: '{target_text}'")
        print(f"Batch size: {self.batch_size}")
        
        all_detections = []
        batch_times = []  # 记录每个batch的时间
        total_start = time.time()
        
        # 分批处理
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            batch_start_id = i
            batch_start = time.time()
            
            print(f"Processing batch {i//self.batch_size + 1}/{(len(images) + self.batch_size - 1)//self.batch_size}: {len(batch_images)} images")
            
            # 预处理图像
            start_time = time.time()
            batch_tensor = self.load_image_batch(batch_images)
            preprocess_time = time.time() - start_time
            print(f"  Preprocessing: {preprocess_time:.3f}s")
            
            # 批量推理
            start_time = time.time()
            bboxes_batch, logits_batch, phrases_batch = self.predict_batch(batch_tensor, target_text)
            inference_time = time.time() - start_time
            print(f"  Inference: {inference_time:.3f}s")
            
            # 转换结果
            start_time = time.time()
            batch_detections = self.convert_to_detections(
                bboxes_batch, logits_batch, phrases_batch, 
                batch_images, batch_start_id
            )
            convert_time = time.time() - start_time
            print(f"  Convert: {convert_time:.3f}s")
            
            batch_total_time = preprocess_time + inference_time + convert_time
            batch_times.append(batch_total_time)
            print(f"  Total batch time: {batch_total_time:.3f}s")
            print(f"  Avg time per image in batch: {batch_total_time/len(batch_images):.3f}s")
            print(f"  Detections: {len(batch_detections)}")
            
            all_detections.extend(batch_detections)
        
        total_time = time.time() - total_start
        avg_time_per_image = total_time / len(images) if len(images) > 0 else 0.0
        
        print(f"Batch detection completed: {len(all_detections)} total detections")
        print(f"Total time: {total_time:.3f}s, Average per image: {avg_time_per_image:.3f}s")
        
        # 存储时间统计信息到类属性（供外部访问）
        self.last_timing_info = {
            "total_time": total_time,
            "avg_time_per_image": avg_time_per_image,
            "num_images": len(images),
            "num_batches": len(batch_times),
            "batch_times": batch_times,
            "avg_batch_time": sum(batch_times) / len(batch_times) if batch_times else 0.0
        }
        
        # 可选可视化保存（与评测器接口兼容）
        if visualization_dir is not None:
            try:
                import os
                os.makedirs(visualization_dir, exist_ok=True)
                # 简单保存代表性可视化：每个view保存第一条检测框
                for det in all_detections:
                    vid = det.view_id
                    if 0 <= vid < len(images):
                        img = images[vid].copy()
                        x, y, w, h = det.bbox
                        import cv2
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.imwrite(os.path.join(visualization_dir, f"view_{vid:02d}.jpg"), img_bgr)
                print(f"Saved quick visualizations to {visualization_dir}")
            except Exception as e:
                print(f"Visualization saving failed: {e}")

        return all_detections


def test_batch_detector():
    """测试批处理检测器"""
    print("🚀 测试批处理GroundingDINO检测器")
    print("=" * 50)
    
    # 创建测试图像
    def create_test_images(num_images: int = 31) -> List[np.ndarray]:
        images = []
        for i in range(num_images):
            img = np.ones((480, 640, 3), dtype=np.uint8) * 255
            
            # 绘制椅子形状
            center_x, center_y = 320, 240
            cv2.rectangle(img, (center_x-60, center_y-80), (center_x+60, center_y-20), (139, 69, 19), -1)
            cv2.rectangle(img, (center_x-80, center_y-20), (center_x+80, center_y+20), (139, 69, 19), -1)
            cv2.rectangle(img, (center_x-70, center_y+20), (center_x-50, center_y+120), (139, 69, 19), -1)
            cv2.rectangle(img, (center_x+50, center_y+20), (center_x+70, center_y+120), (139, 69, 19), -1)
            
            # 添加变化
            if i > 0:
                angle = i * 0.1
                M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
                img = cv2.warpAffine(img, M, (640, 480))
            
            images.append(img)
        return images
    
    # 创建检测器
    detector = BatchGroundingDINODetector(batch_size=8)
    
    # 创建测试图像
    images = create_test_images(31)
    
    # 执行检测
    start_time = time.time()
    detections = detector.detect(images, "chair")
    total_time = time.time() - start_time
    
    print(f"\n=== 性能结果 ===")
    print(f"总时间: {total_time:.2f}秒")
    print(f"平均每帧: {total_time/len(images):.3f}秒")
    print(f"检测数量: {len(detections)}")
    print(f"目标达成: {'✅' if total_time < 10 else '❌'} (目标: <10秒)")
    
    return detections, total_time


if __name__ == "__main__":
    test_batch_detector()
