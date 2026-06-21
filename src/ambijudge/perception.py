"""
4.2 Perception Engine: From Pixels to Evidence
==============================================

Training-free perception engine that:
1. Parses natural language instructions
2. Detects and unifies object instances across multiple views
3. Selects representative images for each instance

This module implements the core perception pipeline without any trainable parameters.
"""
from __future__ import annotations
import numpy as np
import torch
import cv2
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import spacy
import json
import math
import time
from pathlib import Path

@dataclass
class CameraParams:
    """Camera intrinsic and extrinsic parameters"""
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray

@dataclass
class Detection:
    """Single object detection result"""
    view_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    mask: Optional[np.ndarray] = None

@dataclass
class ParsedInstruction:
    """Structured instruction after NLP parsing"""
    target: str
    attributes: List[str]
    relations: str
    action: Optional[str] = None

@dataclass
class InstanceCandidate:
    """Unified object instance candidate"""
    id: str
    representative_image: str
    representative_bbox: Tuple[int, int, int, int]
    detections: List[Detection]
    score: float

class InstructionParser:
    """4.2.1 Instruction Deconstruction using traditional NLP"""

    def __init__(self):
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            raise RuntimeError('Please install spaCy English model: python -m spacy download en_core_web_sm')
        self.attribute_synonyms = {'red': ['crimson', 'scarlet', 'ruby'], 'blue': ['azure', 'navy', 'cobalt'], 'green': ['emerald', 'lime', 'forest'], 'yellow': ['golden', 'amber', 'lemon'], 'black': ['dark', 'ebony'], 'white': ['pale', 'ivory', 'snow'], 'big': ['large', 'huge', 'giant'], 'small': ['tiny', 'little', 'mini'], 'round': ['circular', 'spherical'], 'square': ['rectangular', 'boxy']}
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

    def parse(self, instruction: str) -> ParsedInstruction:
        """
        Parse natural language instruction into structured components
        
        Args:
            instruction: Natural language instruction (e.g., "put the red cup on the tray")
            
        Returns:
            ParsedInstruction with target, attributes, relations, action
        """
        doc = self.nlp(instruction.lower())
        target = None
        for token in doc:
            if token.pos_ == 'NOUN' and token.text not in self.stop_words:
                target = token.text
                break
        if target is None:
            for chunk in doc.noun_chunks:
                if chunk.root.pos_ == 'NOUN':
                    target = chunk.root.text
                    break
        attributes = []
        for token in doc:
            if token.pos_ == 'ADJ' and token.text not in self.stop_words:
                attr = token.text
                for key, synonyms in self.attribute_synonyms.items():
                    if attr in synonyms or attr == key:
                        attr = key
                        break
                attributes.append(attr)
        relations = []
        for token in doc:
            if token.dep_ == 'prep':
                relations.append(token.text + ' ' + ' '.join([child.text for child in token.children]))
        relations_str = ' '.join(relations) if relations else ''
        action = None
        for token in doc:
            if token.pos_ == 'VERB':
                action = token.lemma_
                break
        return ParsedInstruction(target=target or 'object', attributes=attributes, relations=relations_str, action=action)

class InstanceUnifier:
    """4.2.2 Instance Unification and Counting using geometric consistency"""

    def __init__(self, use_hvim: bool=True, voxel_size: float=0.05, K_max: int=16, tau_iou: float=0.25, tau_iom: float=0.6, min_detections: int=2, use_gpu: bool=True, max_memory_gb: float=4.0, scannet_root: Optional[str]=None, neighbor_threshold: int=10, distance_threshold: float=0.3, scale_threshold: float=0.2, angle_min: float=0.0, angle_max: float=60.0, single_conf_threshold: float=0.3, area_threshold: float=0.01, boundary_pixels: int=4):
        """
        Initialize instance unifier with H-VIM and geometric parameters
        
        Args:
            use_hvim: Enable H-VIM GPU-accelerated merging
            voxel_size: Voxel size in meters for H-VIM
            K_max: Maximum detections per voxel (hot voxel truncation)
            tau_iou: IoU threshold for merging
            tau_iom: IoM threshold for merging
            min_detections: Minimum detections per instance
            use_gpu: Enable GPU acceleration for H-VIM
            max_memory_gb: Maximum GPU memory to use
            neighbor_threshold: Maximum view difference for pairing (Δ) - fallback
            distance_threshold: Maximum ray distance for merging (τ_d) - fallback
            scale_threshold: Minimum scale ratio for pairing (τ_scale) - fallback
            angle_min: Minimum ray angle for pairing (θ_min) - fallback
            angle_max: Maximum ray angle for pairing (θ_max) - fallback
            single_conf_threshold: Confidence threshold for single detections - fallback
            area_threshold: Minimum area ratio for keeping groups - fallback
            boundary_pixels: Boundary distance threshold (δ) - fallback
        """
        self.use_hvim = use_hvim
        self.voxel_size = voxel_size
        self.K_max = K_max
        self.tau_iou = tau_iou
        self.tau_iom = tau_iom
        self.min_detections = min_detections
        self.use_gpu = use_gpu
        self.max_memory_gb = max_memory_gb
        self.scannet_root = scannet_root
        self.neighbor_threshold = neighbor_threshold
        self.distance_threshold = distance_threshold
        self.scale_threshold = scale_threshold
        self.angle_min = math.radians(angle_min)
        self.angle_max = math.radians(angle_max)
        self.single_conf_threshold = single_conf_threshold
        self.area_threshold = area_threshold
        self.boundary_pixels = boundary_pixels

    def _ray_from_detection(self, detection: Detection, camera: CameraParams) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert detection to 3D ray in world coordinates
        
        Args:
            detection: Detection result
            camera: Camera parameters
            
        Returns:
            Tuple of (ray_origin, ray_direction)
        """
        x, y, w, h = detection.bbox
        u = x + w / 2
        v = y + h / 2
        u_homogeneous = np.array([u, v, 1.0])
        p_cam = np.linalg.inv(camera.K) @ u_homogeneous
        p_world = camera.R.T @ p_cam
        p_world = p_world / np.linalg.norm(p_world)
        o_world = -camera.R.T @ camera.t
        r_world = camera.R.T @ p_world
        r_world = r_world / np.linalg.norm(r_world)
        return (o_world, r_world)

    def _ray_distance(self, ray1: Tuple[np.ndarray, np.ndarray], ray2: Tuple[np.ndarray, np.ndarray]) -> float:
        """
        Calculate minimum distance between two rays
        
        Args:
            ray1: (origin1, direction1)
            ray2: (origin2, direction2)
            
        Returns:
            Minimum distance between rays
        """
        o1, r1 = ray1
        o2, r2 = ray2
        w0 = o1 - o2
        b = np.dot(r1, r2)
        d = np.dot(r1, w0)
        e = np.dot(r2, w0)
        if abs(1 - b ** 2) < 1e-08:
            return np.linalg.norm(np.cross(r1, w0))
        lambda1 = (b * e - d) / (1 - b ** 2)
        lambda2 = (e - b * d) / (1 - b ** 2)
        p1 = o1 + lambda1 * r1
        p2 = o2 + lambda2 * r2
        return np.linalg.norm(p1 - p2)

    def _angle_between_rays(self, ray1: Tuple[np.ndarray, np.ndarray], ray2: Tuple[np.ndarray, np.ndarray]) -> float:
        """Calculate angle between two rays"""
        _, r1 = ray1
        _, r2 = ray2
        cos_angle = np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return math.acos(cos_angle)

    def _scale_ratio(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate scale ratio between two bounding boxes"""
        area1 = bbox1[2] * bbox1[3]
        area2 = bbox2[2] * bbox2[3]
        return min(area1, area2) / max(area1, area2)

    def _is_near_boundary(self, bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int]) -> bool:
        """Check if bbox is near image boundary"""
        x, y, w, h = bbox
        h_img, w_img = image_shape[:2]
        return x < self.boundary_pixels or y < self.boundary_pixels or x + w > w_img - self.boundary_pixels or (y + h > h_img - self.boundary_pixels)

    def unify_instances(self, detections: List[Detection], cameras: List[CameraParams], images: List[np.ndarray]) -> List[InstanceCandidate]:
        """
        H-VIM-based instance unification using GPU-accelerated voxel merging
        
        Args:
            detections: List of all detections across views
            cameras: Camera parameters for each view
            images: Images for each view
            
        Returns:
            List of unified instance candidates
        """
        if not detections:
            print('No detections to unify')
            return []
        print(f'H-VIM: Starting instance unification with {len(detections)} detections')
        if not self.use_hvim:
            print('H-VIM: Disabled, using geometric method')
            return self._unify_instances_geometric(detections, cameras, images)
        try:
            from ambijudge.hvim import HvimUnifier as HVIMUnifier
            scene_path = self._get_scene_path(detections[0])
            if scene_path is None:
                print('H-VIM: No scene path available, falling back to geometric method')
                return self._unify_instances_geometric(detections, cameras, images)
            hvim = HVIMUnifier(voxel_size=self.voxel_size, K_max=self.K_max, tau_iou=self.tau_iou, tau_iom=self.tau_iom, min_detections=self.min_detections, use_gpu=self.use_gpu, max_memory_gb=self.max_memory_gb)
            instance_dicts = hvim.unify(detections, cameras, scene_path)
            candidates = []
            for inst_dict in instance_dicts:
                candidate = InstanceCandidate(id=inst_dict['id'], representative_image=inst_dict['representative_image'], representative_bbox=inst_dict['representative_bbox'], detections=inst_dict['detections'], score=inst_dict['score'])
                candidates.append(candidate)
            print(f'H-VIM: Completed with {len(candidates)} instances')
            return candidates
        except Exception as e:
            print(f'H-VIM failed: {e}, falling back to geometric method')
            return self._unify_instances_geometric(detections, cameras, images)

    def _get_scene_path(self, detection: Detection) -> Optional[str]:
        """
        Extract scene path from detection metadata or global context
        
        Args:
            detection: Detection object
            
        Returns:
            Scene path string or None
        """
        if hasattr(detection, 'metadata') and 'scene_path' in detection.metadata:
            return detection.metadata['scene_path']
        if hasattr(detection, 'scene_path'):
            return detection.scene_path
        if self.scannet_root is not None:
            scene_id = 'scene0000_00'
            if hasattr(detection, 'metadata') and 'scene_id' in detection.metadata:
                scene_id = detection.metadata['scene_id']
            return str(Path(self.scannet_root) / scene_id)
        return None

    def _unify_instances_geometric(self, detections: List[Detection], cameras: List[CameraParams], images: List[np.ndarray]) -> List[InstanceCandidate]:
        """
        Fallback geometric instance unification (original method)
        
        Args:
            detections: List of all detections across views
            cameras: Camera parameters for each view
            images: Images for each view
            
        Returns:
            List of unified instance candidates
        """
        print('Using optimized fast merging method...')
        return self._unify_instances_fast(detections, cameras, images)

    def _unify_instances_fast(self, detections: List[Detection], cameras: List[CameraParams], images: List[np.ndarray]) -> List[InstanceCandidate]:
        if not detections:
            return []
        detections_by_view = defaultdict(list)
        for i, det in enumerate(detections):
            detections_by_view[det.view_id].append((i, det))
        print(f'Detections per view: {dict(sorted(detections_by_view.items()))}')
        view_groups = {}
        for view_id, view_detections in detections_by_view.items():
            if len(view_detections) <= 1:
                view_groups[view_id] = [view_detections]
                continue
            groups = self._merge_by_iou_same_view(view_detections)
            view_groups[view_id] = groups
        cross_view_groups = self._merge_cross_views_fast(view_groups, detections, cameras)
        candidates = []
        for group_id, group_detections in enumerate(cross_view_groups):
            if not group_detections:
                continue
            representative = self._select_best_representative(group_detections, detections_by_view)
            score = self._calculate_instance_score(group_detections)
            candidate = InstanceCandidate(id=f'I_{len(candidates) + 1}', representative_image=f'view_{representative.view_id:03d}.jpg', representative_bbox=representative.bbox, detections=[det for _, det in group_detections], score=score)
            candidates.append(candidate)
            print(f'  -> Created instance {candidate.id} with {len(group_detections)} detections, score={score:.4f}')
        print(f'Final result: {len(candidates)} unified instances from {len(detections)} original detections')
        return candidates

    def _merge_by_iou_same_view(self, view_detections):
        if len(view_detections) <= 1:
            return [view_detections]
        groups = []
        used = set()
        for i, (idx1, det1) in enumerate(view_detections):
            if i in used:
                continue
            current_group = [(idx1, det1)]
            used.add(i)
            for j, (idx2, det2) in enumerate(view_detections[i + 1:], i + 1):
                if j in used:
                    continue
                iou = self._calculate_iou(det1.bbox, det2.bbox)
                if iou > 0.2:
                    current_group.append((idx2, det2))
                    used.add(j)
            groups.append(current_group)
        return groups

    def _merge_cross_views_fast(self, view_groups, detections, cameras):
        parent = list(range(len(detections)))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = (find(x), find(y))
            if px != py:
                parent[px] = py
        view_ids = sorted(view_groups.keys())
        unions_made = 0
        for i, view_id in enumerate(view_ids):
            for group in view_groups[view_id]:
                if not group:
                    continue
                for neighbor_idx in range(max(0, i - 3), min(len(view_ids), i + 4)):
                    if neighbor_idx == i:
                        continue
                    neighbor_view_id = view_ids[neighbor_idx]
                    for neighbor_group in view_groups[neighbor_view_id]:
                        if not neighbor_group:
                            continue
                        if self._can_merge_groups(group, neighbor_group, detections, cameras):
                            for idx1, _ in group:
                                for idx2, _ in neighbor_group:
                                    union(idx1, idx2)
                            unions_made += 1
                            break
        print(f'Made {unions_made} cross-view unions')
        groups = defaultdict(list)
        for i, det in enumerate(detections):
            root = find(i)
            groups[root].append((i, det))
        return list(groups.values())

    def _can_merge_groups(self, group1, group2, detections, cameras):
        if not group1 or not group2:
            return False
        rep1 = max(group1, key=lambda x: x[1].confidence)[1]
        rep2 = max(group2, key=lambda x: x[1].confidence)[1]
        scale_ratio = self._scale_ratio(rep1.bbox, rep2.bbox)
        if scale_ratio < 0.2:
            return False
        try:
            ray1 = self._ray_from_detection(rep1, cameras[rep1.view_id])
            ray2 = self._ray_from_detection(rep2, cameras[rep2.view_id])
            distance = self._ray_distance(ray1, ray2)
            if distance < 0.8:
                return True
        except:
            pass
        return False

    def _select_best_representative(self, group_detections, detections_by_view):
        view_counts = defaultdict(int)
        for idx, det in group_detections:
            view_counts[det.view_id] += 1
        best_view = max(view_counts.keys(), key=lambda v: view_counts[v])
        best_detection = None
        best_confidence = 0
        for idx, det in group_detections:
            if det.view_id == best_view and det.confidence > best_confidence:
                best_detection = det
                best_confidence = det.confidence
        return best_detection

    def _calculate_instance_score(self, group_detections):
        if not group_detections:
            return 0.0
        total_area = sum((det.bbox[2] * det.bbox[3] for _, det in group_detections))
        if total_area == 0:
            return sum((det.confidence for _, det in group_detections)) / len(group_detections)
        weighted_score = sum((det.confidence * (det.bbox[2] * det.bbox[3]) for _, det in group_detections)) / total_area
        return weighted_score

    def _calculate_iou(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        return intersection / union if union > 0 else 0.0
        rays = []
        for i, det in enumerate(detections):
            if det.view_id < len(cameras):
                ray = self._ray_from_detection(det, cameras[det.view_id])
                rays.append((i, ray))
        print(f'Generated {len(rays)} rays for geometric analysis')
        parent = list(range(len(detections)))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = (find(x), find(y))
            if px != py:
                parent[px] = py
        unions_made = 0
        total_pairs_checked = 0
        detections_by_view = defaultdict(list)
        for i, det in enumerate(detections):
            detections_by_view[det.view_id].append((i, det))
        for view_id in sorted(detections_by_view.keys()):
            current_detections = detections_by_view[view_id]
            max_view_id = max(detections_by_view.keys()) if detections_by_view else 0
            for neighbor_view in range(max(0, view_id - self.neighbor_threshold), min(max_view_id + 1, view_id + self.neighbor_threshold + 1)):
                if neighbor_view == view_id:
                    continue
                if neighbor_view in detections_by_view:
                    neighbor_detections = detections_by_view[neighbor_view]
                    for i, (idx1, det1) in enumerate(current_detections):
                        for j, (idx2, det2) in enumerate(neighbor_detections):
                            if idx1 >= idx2:
                                continue
                            total_pairs_checked += 1
                            ray1 = rays[idx1][1]
                            ray2 = rays[idx2][1]
                scale_ratio = self._scale_ratio(det1.bbox, det2.bbox)
                if scale_ratio < self.scale_threshold:
                    continue
                angle = self._angle_between_rays(ray1, ray2)
                if not self.angle_min <= angle <= self.angle_max:
                    continue
                distance = self._ray_distance(ray1, ray2)
                if distance < self.distance_threshold:
                    union(idx1, idx2)
                    unions_made += 1
        print(f'Checked {total_pairs_checked} detection pairs, made {unions_made} unions')
        print(f'Thresholds: neighbor={self.neighbor_threshold}, distance={self.distance_threshold}, scale={self.scale_threshold}')
        print(f'Angle range: {self.angle_min:.1f}°-{self.angle_max:.1f}°, single_conf={self.single_conf_threshold}')
        groups = defaultdict(list)
        for i, det in enumerate(detections):
            root = find(i)
            groups[root].append((i, det))
        print(f'Formed {len(groups)} instance groups')
        candidates = []
        for group_id, group in groups.items():
            if not group:
                continue
            print(f'Processing group {group_id} with {len(group)} detections')
            if len(group) == 1:
                _, det = group[0]
                if det.confidence < self.single_conf_threshold:
                    continue
            confidences = [det.confidence for _, det in group]
            areas = [det.bbox[2] * det.bbox[3] for _, det in group]
            total_area = sum(areas)
            if total_area < self.area_threshold * (images[0].shape[0] * images[0].shape[1]):
                continue
            best_det = None
            best_score = -1
            for _, det in group:
                if det.mask is not None:
                    visibility = np.sum(det.mask) / (det.bbox[2] * det.bbox[3])
                else:
                    img_area = images[det.view_id].shape[0] * images[det.view_id].shape[1]
                    visibility = det.bbox[2] * det.bbox[3] / img_area
                boundary_penalty = 0.5 if self._is_near_boundary(det.bbox, images[det.view_id].shape) else 1.0
                score = det.confidence * visibility * boundary_penalty
                if score > best_score:
                    best_score = score
                    best_det = det
            if best_det is not None:
                candidate = InstanceCandidate(id=f'I_{len(candidates) + 1}', representative_image=f'view_{best_det.view_id:02d}.jpg', representative_bbox=best_det.bbox, detections=[det for _, det in group], score=best_score)
                candidates.append(candidate)
                print(f'  -> Created instance {candidate.id} with {len(group)} detections, score={best_score:.4f}')
        print(f'Final result: {len(candidates)} unified instances from {len(detections)} original detections')
        return candidates

class PerceptionEngine:
    """Main perception engine combining instruction parsing and instance unification"""

    def __init__(self, **kwargs):
        """
        Initialize perception engine with configurable parameters
        
        Args:
            **kwargs: Parameters for InstanceUnifier
        """
        self.parser = InstructionParser()
        self.unifier = InstanceUnifier(**kwargs)

    def process(self, instruction: str, images: List[np.ndarray], cameras: List[CameraParams], detections: List[Detection]) -> Tuple[ParsedInstruction, List[InstanceCandidate]]:
        """
        Main processing pipeline: parse instruction and unify instances
        
        Args:
            instruction: Natural language instruction
            images: List of input images
            cameras: Camera parameters for each view
            detections: Object detections from DINO-X API
            
        Returns:
            Tuple of (parsed_instruction, instance_candidates)
        """
        parsed_instruction = self.parser.parse(instruction)
        unification_start = time.time()
        candidates = self.unifier.unify_instances(detections, cameras, images)
        self.last_unification_time = time.time() - unification_start
        return (parsed_instruction, candidates)
        '\n        Main processing pipeline: parse instruction and unify instances\n        \n        Args:\n            instruction: Natural language instruction\n            images: List of input images\n            cameras: Camera parameters for each view\n            detections: Object detections from DINO-X API\n            \n        Returns:\n            Tuple of (parsed_instruction, instance_candidates)\n        '
        parsed_instruction = self.parser.parse(instruction)
        candidates = self.unifier.unify_instances(detections, cameras, images)
        return (parsed_instruction, candidates)