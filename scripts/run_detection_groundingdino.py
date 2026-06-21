import argparse
import os
import pickle
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from ambijudge.paths import ensure_grounding_dino_on_path, grounding_dino_config_path, grounding_dino_weights_path
GD_ROOT = str(ensure_grounding_dino_on_path())
import cv2
import numpy as np
import torch
from PIL import Image
from groundingdino.util.inference import load_model, load_image, predict, preprocess_caption
from groundingdino.util.utils import get_phrases_from_posmap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--target_text', required=True)
    parser.add_argument('--output_pickle', required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    config_path = str(grounding_dino_config_path())
    weights_path = str(grounding_dino_weights_path())
    model = load_model(config_path, weights_path, device=device)
    box_threshold = 0.5
    text_threshold = 0.5
    images_dir = Path(args.images_dir)
    image_files = sorted(images_dir.glob('view_*.jpg'))
    if not image_files:
        image_files = sorted(images_dir.glob('*.jpg'))
    results = []
    caption = preprocess_caption(args.target_text)
    for view_id, img_path in enumerate(image_files):
        try:
            image_source, image_tensor = load_image(str(img_path))
            h, w = image_source.shape[:2]
            with torch.no_grad():
                outputs = model(image_tensor[None].to(device), captions=[caption])
            prediction_logits = outputs['pred_logits'].cpu().sigmoid()[0]
            prediction_boxes = outputs['pred_boxes'].cpu()[0]
            mask = prediction_logits.max(dim=1)[0] > box_threshold
            if mask.sum() == 0:
                continue
            logits = prediction_logits[mask]
            boxes = prediction_boxes[mask]
            tokenizer = model.tokenizer
            tokenized = tokenizer(caption)
            for bbox, logit in zip(boxes, logits):
                conf = float(logit.max())
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
                results.append({'view_id': view_id, 'bbox': (x1, y1, x2 - x1, y2 - y1), 'confidence': conf})
        except Exception as e:
            print(f'Warning: detection failed for {img_path}: {e}', file=sys.stderr)
    with open(args.output_pickle, 'wb') as f:
        pickle.dump(results, f)
if __name__ == '__main__':
    main()