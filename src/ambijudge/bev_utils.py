"""Load user-generated BEV maps (see docs/BEV.md)."""
from __future__ import annotations
import os
import cv2
from .paths import bev_maps_dir
SUPPORTED_EXTS = ['.png', '.jpg', '.jpeg']

def _find_file(base_dir, base_name):
    for ext in SUPPORTED_EXTS:
        path = os.path.join(base_dir, base_name + ext)
        if os.path.isfile(path):
            return path
    return None

def generate_bev_from_depth_scene(scannet_root, scene_id, bev_dir=None, **kwargs):
    """
    Load precomputed bird's-eye view images for a ScanNet scene.

    Generate maps first with BundleFusion — see docs/BEV.md.
    Expected files under bev_maps/ (override with BEV_MAPS_DIR):
      - grayscale: {scene_id}.(png|jpg)
      - color: {scene_id}_color.(png|jpg)
    """
    del scannet_root, kwargs
    bev_root = str(bev_dir or bev_maps_dir())
    gray_path = _find_file(bev_root, f'{scene_id}')
    color_path = _find_file(bev_root, f'{scene_id}_color')
    if gray_path is None and color_path is None:
        raise FileNotFoundError(f'BEV image not found: {bev_root}/{scene_id}.(png|jpg) or {scene_id}_color.(png|jpg). Set BEV_MAPS_DIR or place maps under ./bev_maps/.')
    gray_img = cv2.imread(gray_path, cv2.IMREAD_UNCHANGED) if gray_path else None
    color_img = cv2.imread(color_path, cv2.IMREAD_UNCHANGED) if color_path else None
    if gray_img is None and color_img is not None:
        if color_img.ndim == 3:
            gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = color_img.copy()
    if color_img is None and gray_img is not None:
        if gray_img.ndim == 2:
            color_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
        else:
            color_img = gray_img.copy()
    if gray_img is None or color_img is None:
        raise FileNotFoundError(f'Incomplete BEV image for {scene_id}: gray={gray_path}, color={color_path}')
    return (gray_img, color_img)