import numpy as np
import math
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation as R
from .perception import CameraParams

class KeyframeFilter:

    def __init__(self, translation_threshold: float=0.05, rotation_threshold: float=5.0):
        self.translation_threshold = translation_threshold
        self.rotation_threshold = rotation_threshold
        self.last_pose = None

    def filter_keyframes(self, images: List[np.ndarray], cameras: List[CameraParams]) -> Tuple[List[np.ndarray], List[CameraParams], List[int]]:
        if len(images) != len(cameras):
            raise ValueError(f'Images and cameras count mismatch: {len(images)} vs {len(cameras)}')
        if len(images) == 0:
            return ([], [], [])
        keyframe_images = []
        keyframe_cameras = []
        keyframe_indices = []
        keyframe_images.append(images[0])
        keyframe_cameras.append(cameras[0])
        keyframe_indices.append(0)
        self.last_pose = self._extract_pose(cameras[0])
        print(f'Keyframe filtering: Starting with {len(images)} frames')
        print(f'Thresholds: translation={self.translation_threshold}m, rotation={self.rotation_threshold}°')
        for i in range(1, len(images)):
            current_pose = self._extract_pose(cameras[i])
            translation_diff, rotation_diff = self._compute_pose_difference(self.last_pose, current_pose)
            if translation_diff >= self.translation_threshold or rotation_diff >= self.rotation_threshold:
                keyframe_images.append(images[i])
                keyframe_cameras.append(cameras[i])
                keyframe_indices.append(i)
                self.last_pose = current_pose
                print(f'  Frame {i:3d}: Keep (trans={translation_diff:.3f}m, rot={rotation_diff:.1f}°)')
            else:
                print(f'  Frame {i:3d}: Skip (trans={translation_diff:.3f}m, rot={rotation_diff:.1f}°)')
        print(f'Keyframe filtering completed: {len(images)} -> {len(keyframe_images)} frames')
        print(f'Reduction ratio: {len(keyframe_images) / len(images) * 100:.1f}%')
        return (keyframe_images, keyframe_cameras, keyframe_indices)

    def _extract_pose(self, camera: CameraParams) -> Tuple[np.ndarray, np.ndarray]:
        position = -camera.R.T @ camera.t
        rotation_matrix = camera.R
        return (position, rotation_matrix)

    def _compute_pose_difference(self, pose1: Tuple[np.ndarray, np.ndarray], pose2: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, float]:
        pos1, rot1 = pose1
        pos2, rot2 = pose2
        translation_diff = np.linalg.norm(pos2 - pos1)
        relative_rotation = rot2 @ rot1.T
        try:
            r = R.from_matrix(relative_rotation)
            rotation_diff = np.abs(r.as_euler('xyz', degrees=True))
            rotation_diff = np.max(rotation_diff)
        except:
            trace = np.trace(relative_rotation)
            trace = np.clip(trace, -1.0, 1.0)
            rotation_diff = np.arccos((trace - 1) / 2) * 180 / np.pi
        return (translation_diff, rotation_diff)

    def adaptive_filter(self, images: List[np.ndarray], cameras: List[CameraParams], target_frames: int=40) -> Tuple[List[np.ndarray], List[CameraParams], List[int]]:
        if len(images) <= target_frames:
            print(f'Input frames ({len(images)}) <= target frames ({target_frames}), no filtering needed')
            return (images, cameras, list(range(len(images))))
        original_trans_thresh = self.translation_threshold
        original_rot_thresh = self.rotation_threshold
        scale = target_frames / len(images)
        base_trans = 0.3
        base_rot = 20.0
        initial_trans = base_trans * (1 / scale)
        initial_rot = base_rot * (1 / scale)
        initial_trans = np.clip(initial_trans, 0.2, 1.0)
        initial_rot = np.clip(initial_rot, 15.0, 60.0)
        print(f'Quick adaptive filtering: trying trans={initial_trans:.3f}m, rot={initial_rot:.1f}°')
        self.translation_threshold = initial_trans
        self.rotation_threshold = initial_rot
        filtered_images, filtered_cameras, filtered_indices = self.filter_keyframes(images, cameras)
        frame_diff = abs(len(filtered_images) - target_frames)
        if frame_diff <= 5:
            print(f'Quick adaptive filtering: {len(images)} -> {len(filtered_images)} frames (target: {target_frames}, diff: {frame_diff})')
            return (filtered_images, filtered_cameras, filtered_indices)
        print(f'Initial attempt: {len(filtered_images)} frames, diff={frame_diff}, trying binary search...')
        best_result = (filtered_images, filtered_cameras, filtered_indices)
        best_diff = frame_diff
        for iteration in range(3):
            if len(filtered_images) > target_frames:
                self.translation_threshold *= 1.2
                self.rotation_threshold *= 1.2
            else:
                self.translation_threshold *= 0.8
                self.rotation_threshold *= 0.8
            self.translation_threshold = np.clip(self.translation_threshold, 0.2, 1.0)
            self.rotation_threshold = np.clip(self.rotation_threshold, 15.0, 60.0)
            filtered_images, filtered_cameras, filtered_indices = self.filter_keyframes(images, cameras)
            frame_diff = abs(len(filtered_images) - target_frames)
            if frame_diff < best_diff:
                best_diff = frame_diff
                best_result = (filtered_images, filtered_cameras, filtered_indices)
            if frame_diff <= 3:
                break
        self.translation_threshold = original_trans_thresh
        self.rotation_threshold = original_rot_thresh
        print(f'Quick adaptive filtering: {len(images)} -> {len(best_result[0])} frames (target: {target_frames}, final diff: {best_diff})')
        return best_result

    def fast_fixed_filter(self, images: List[np.ndarray], cameras: List[CameraParams], target_frames: int=None) -> Tuple[List[np.ndarray], List[CameraParams], List[int]]:
        filtered_images = []
        filtered_cameras = []
        filtered_indices = []
        filtered_images.append(images[0])
        filtered_cameras.append(cameras[0])
        filtered_indices.append(0)
        last_pose = self._extract_pose(cameras[0])
        for i in range(1, len(images)):
            current_pose = self._extract_pose(cameras[i])
            translation_diff, rotation_diff = self._compute_pose_difference(last_pose, current_pose)
            if translation_diff >= self.translation_threshold or rotation_diff >= self.rotation_threshold:
                filtered_images.append(images[i])
                filtered_cameras.append(cameras[i])
                filtered_indices.append(i)
                last_pose = current_pose
        print(f'Keyframe filtering: {len(images)} -> {len(filtered_images)} frames (thresholds: trans={self.translation_threshold}m, rot={self.rotation_threshold}°)')
        return (filtered_images, filtered_cameras, filtered_indices)

    def _filter_by_poses(self, cameras: List[CameraParams], target_frames: int=None) -> Tuple[List[CameraParams], List[int]]:
        if len(cameras) <= 1:
            return (cameras, [0])
        filtered_cameras = []
        filtered_indices = []
        filtered_cameras.append(cameras[0])
        filtered_indices.append(0)
        last_pose = self._extract_pose(cameras[0])
        for i in range(1, len(cameras)):
            current_pose = self._extract_pose(cameras[i])
            translation_diff, rotation_diff = self._compute_pose_difference(last_pose, current_pose)
            if translation_diff >= self.translation_threshold or rotation_diff >= self.rotation_threshold:
                filtered_cameras.append(cameras[i])
                filtered_indices.append(i)
                last_pose = current_pose
        return (filtered_cameras, filtered_indices)

def create_keyframe_filter(translation_threshold: float=0.15, rotation_threshold: float=15.0) -> KeyframeFilter:
    return KeyframeFilter(translation_threshold, rotation_threshold)