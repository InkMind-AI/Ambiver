"""
Keyframe Filter for ScanNet Dataset
===================================

基于相机姿态的关键帧筛选算法，用于减少冗余图像数量，
保留具有显著视角变化的图像，目标保留约40张图像。
"""

import numpy as np
import math
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation as R
from .perception import CameraParams


class KeyframeFilter:
    """
    基于相机姿态的关键帧筛选器
    
    核心逻辑：
    1. 计算相机位置和旋转的变化
    2. 设置平移和旋转阈值
    3. 只保留满足阈值的帧作为关键帧
    """
    
    def __init__(self, 
                 translation_threshold: float = 0.05,  # 5cm平移阈值（调低以保留更多帧）
                 rotation_threshold: float = 5.0):    # 5度旋转阈值（调低以保留更多帧）
        """
        初始化关键帧筛选器
        
        Args:
            translation_threshold: 平移阈值（米），默认15cm
            rotation_threshold: 旋转阈值（度），默认15度
        """
        self.translation_threshold = translation_threshold
        self.rotation_threshold = rotation_threshold
        self.last_pose = None
        
    def filter_keyframes(self, 
                        images: List[np.ndarray], 
                        cameras: List[CameraParams]) -> Tuple[List[np.ndarray], List[CameraParams], List[int]]:
        """
        基于相机姿态筛选关键帧
        
        Args:
            images: 原始图像列表
            cameras: 对应的相机参数列表
            
        Returns:
            (筛选后的图像列表, 筛选后的相机参数列表, 保留的帧索引列表)
        """
        if len(images) != len(cameras):
            raise ValueError(f"Images and cameras count mismatch: {len(images)} vs {len(cameras)}")
        
        if len(images) == 0:
            return [], [], []
        
        # 初始化关键帧列表
        keyframe_images = []
        keyframe_cameras = []
        keyframe_indices = []
        
        # 第一帧无条件保留
        keyframe_images.append(images[0])
        keyframe_cameras.append(cameras[0])
        keyframe_indices.append(0)
        
        # 记录第一帧的相机姿态
        self.last_pose = self._extract_pose(cameras[0])
        
        print(f"Keyframe filtering: Starting with {len(images)} frames")
        print(f"Thresholds: translation={self.translation_threshold}m, rotation={self.rotation_threshold}°")
        
        # 遍历后续帧
        for i in range(1, len(images)):
            current_pose = self._extract_pose(cameras[i])
            
            # 计算与上一关键帧的差异
            translation_diff, rotation_diff = self._compute_pose_difference(self.last_pose, current_pose)
            
            # 判断是否满足阈值条件
            if (translation_diff >= self.translation_threshold or 
                rotation_diff >= self.rotation_threshold):
                
                # 满足条件，保留为关键帧
                keyframe_images.append(images[i])
                keyframe_cameras.append(cameras[i])
                keyframe_indices.append(i)
                
                # 更新last_pose
                self.last_pose = current_pose
                
                print(f"  Frame {i:3d}: Keep (trans={translation_diff:.3f}m, rot={rotation_diff:.1f}°)")
            else:
                print(f"  Frame {i:3d}: Skip (trans={translation_diff:.3f}m, rot={rotation_diff:.1f}°)")
        
        print(f"Keyframe filtering completed: {len(images)} -> {len(keyframe_images)} frames")
        print(f"Reduction ratio: {len(keyframe_images)/len(images)*100:.1f}%")
        
        return keyframe_images, keyframe_cameras, keyframe_indices
    
    def _extract_pose(self, camera: CameraParams) -> Tuple[np.ndarray, np.ndarray]:
        """
        从相机参数提取姿态（位置和旋转）
        
        Args:
            camera: 相机参数对象
            
        Returns:
            (position, rotation_matrix) 元组
        """
        # 从相机外参计算世界坐标系下的位置
        # 相机位置 = -R^T * t
        position = -camera.R.T @ camera.t
        
        # 旋转矩阵
        rotation_matrix = camera.R
        
        return position, rotation_matrix
    
    def _compute_pose_difference(self, 
                                pose1: Tuple[np.ndarray, np.ndarray], 
                                pose2: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, float]:
        """
        计算两个姿态之间的差异
        
        Args:
            pose1: 第一个姿态 (position, rotation_matrix)
            pose2: 第二个姿态 (position, rotation_matrix)
            
        Returns:
            (translation_difference, rotation_difference) 元组
        """
        pos1, rot1 = pose1
        pos2, rot2 = pose2
        
        # 计算平移差异（欧几里得距离）
        translation_diff = np.linalg.norm(pos2 - pos1)
        
        # 计算旋转差异（角度）
        # 使用相对旋转矩阵 R_rel = R2 * R1^T
        relative_rotation = rot2 @ rot1.T
        
        # 从旋转矩阵提取旋转角度
        # 使用scipy的Rotation类来处理旋转
        try:
            r = R.from_matrix(relative_rotation)
            rotation_diff = np.abs(r.as_euler('xyz', degrees=True))
            # 取最大旋转角度分量
            rotation_diff = np.max(rotation_diff)
        except:
            # 备用方法：使用旋转矩阵的迹
            trace = np.trace(relative_rotation)
            # 限制在[-1, 1]范围内，避免数值误差
            trace = np.clip(trace, -1.0, 1.0)
            rotation_diff = np.arccos((trace - 1) / 2) * 180 / np.pi
        
        return translation_diff, rotation_diff
    
    def adaptive_filter(self, 
                       images: List[np.ndarray], 
                       cameras: List[CameraParams],
                       target_frames: int = 40) -> Tuple[List[np.ndarray], List[CameraParams], List[int]]:
        """
        快速自适应关键帧筛选，自动调整阈值以达到目标帧数
        
        Args:
            images: 原始图像列表
            cameras: 对应的相机参数列表
            target_frames: 目标保留帧数，默认40
            
        Returns:
            (筛选后的图像列表, 筛选后的相机参数列表, 保留的帧索引列表)
        """
        if len(images) <= target_frames:
            print(f"Input frames ({len(images)}) <= target frames ({target_frames}), no filtering needed")
            return images, cameras, list(range(len(images)))
        
        # 保存原始阈值
        original_trans_thresh = self.translation_threshold
        original_rot_thresh = self.rotation_threshold
        
        # 快速策略：基于比例直接计算初始阈值
        scale = target_frames / len(images)
        base_trans = 0.30  # 更严格的基础平移阈值
        base_rot = 20.0    # 更严格的基础旋转阈值
        
        # 计算初始阈值
        initial_trans = base_trans * (1 / scale)
        initial_rot = base_rot * (1 / scale)
        
        # 限制阈值范围（更严格）
        initial_trans = np.clip(initial_trans, 0.20, 1.0)
        initial_rot = np.clip(initial_rot, 15.0, 60.0)
        
        print(f"Quick adaptive filtering: trying trans={initial_trans:.3f}m, rot={initial_rot:.1f}°")
        
        # 尝试初始阈值
        self.translation_threshold = initial_trans
        self.rotation_threshold = initial_rot
        
        filtered_images, filtered_cameras, filtered_indices = self.filter_keyframes(images, cameras)
        frame_diff = abs(len(filtered_images) - target_frames)
        
        # 如果已经很接近目标（差异<=5），直接返回
        if frame_diff <= 5:
            print(f"Quick adaptive filtering: {len(images)} -> {len(filtered_images)} frames (target: {target_frames}, diff: {frame_diff})")
            return filtered_images, filtered_cameras, filtered_indices
        
        # 如果差异较大，进行快速二分搜索（最多3次迭代）
        print(f"Initial attempt: {len(filtered_images)} frames, diff={frame_diff}, trying binary search...")
        
        best_result = (filtered_images, filtered_cameras, filtered_indices)
        best_diff = frame_diff
        
        # 快速二分搜索
        for iteration in range(3):
            if len(filtered_images) > target_frames:
                # 保留的帧太多，增加阈值（更严格）
                self.translation_threshold *= 1.2
                self.rotation_threshold *= 1.2
            else:
                # 保留的帧太少，减少阈值（更宽松）
                self.translation_threshold *= 0.8
                self.rotation_threshold *= 0.8
            
            # 限制阈值范围（更严格）
            self.translation_threshold = np.clip(self.translation_threshold, 0.20, 1.0)
            self.rotation_threshold = np.clip(self.rotation_threshold, 15.0, 60.0)
            
            # 执行筛选
            filtered_images, filtered_cameras, filtered_indices = self.filter_keyframes(images, cameras)
            frame_diff = abs(len(filtered_images) - target_frames)
            
            if frame_diff < best_diff:
                best_diff = frame_diff
                best_result = (filtered_images, filtered_cameras, filtered_indices)
            
            # 如果已经很接近目标，提前结束
            if frame_diff <= 3:
                break
        
        # 恢复原始阈值
        self.translation_threshold = original_trans_thresh
        self.rotation_threshold = original_rot_thresh
        
        print(f"Quick adaptive filtering: {len(images)} -> {len(best_result[0])} frames (target: {target_frames}, final diff: {best_diff})")
        return best_result
    
    def fast_fixed_filter(self, 
                         images: List[np.ndarray], 
                         cameras: List[CameraParams],
                         target_frames: int = None) -> Tuple[List[np.ndarray], List[CameraParams], List[int]]:
        """
        快速固定阈值筛选，O(n)时间复杂度
        完全由超参数控制，不强制限制帧数
        
        Args:
            images: 原始图像列表
            cameras: 对应的相机参数列表
            target_frames: 目标帧数（已废弃，仅用于兼容性）
            
        Returns:
            (filtered_images, filtered_cameras, filtered_indices)
        """
        # 使用固定阈值进行一次性筛选
        filtered_images = []
        filtered_cameras = []
        filtered_indices = []
        
        # 总是保留第一帧
        filtered_images.append(images[0])
        filtered_cameras.append(cameras[0])
        filtered_indices.append(0)
        
        last_pose = self._extract_pose(cameras[0])
        
        for i in range(1, len(images)):
            current_pose = self._extract_pose(cameras[i])
            
            # 计算姿态差异
            translation_diff, rotation_diff = self._compute_pose_difference(last_pose, current_pose)
            
            # 检查是否满足阈值
            if (translation_diff >= self.translation_threshold or 
                rotation_diff >= self.rotation_threshold):
                
                filtered_images.append(images[i])
                filtered_cameras.append(cameras[i])
                filtered_indices.append(i)
                last_pose = current_pose
        
        print(f"Keyframe filtering: {len(images)} -> {len(filtered_images)} frames (thresholds: trans={self.translation_threshold}m, rot={self.rotation_threshold}°)")
        return filtered_images, filtered_cameras, filtered_indices
    
    def _filter_by_poses(self, cameras: List[CameraParams], target_frames: int = None) -> Tuple[List[CameraParams], List[int]]:
        """
        仅基于相机参数进行筛选，不涉及图像加载
        用于优化：先筛选再加载图像
        
        Args:
            cameras: 相机参数列表
            target_frames: 目标帧数（已废弃，仅用于兼容性）
            
        Returns:
            (filtered_cameras, filtered_indices)
        """
        if len(cameras) <= 1:
            return cameras, [0]
        
        filtered_cameras = []
        filtered_indices = []
        
        # 总是保留第一帧
        filtered_cameras.append(cameras[0])
        filtered_indices.append(0)
        
        last_pose = self._extract_pose(cameras[0])
        
        for i in range(1, len(cameras)):
            current_pose = self._extract_pose(cameras[i])
            
            # 计算姿态差异
            translation_diff, rotation_diff = self._compute_pose_difference(last_pose, current_pose)
            
            # 检查是否满足阈值
            if (translation_diff >= self.translation_threshold or 
                rotation_diff >= self.rotation_threshold):
                
                filtered_cameras.append(cameras[i])
                filtered_indices.append(i)
                last_pose = current_pose
        
        return filtered_cameras, filtered_indices


def create_keyframe_filter(translation_threshold: float = 0.15, 
                          rotation_threshold: float = 15.0) -> KeyframeFilter:
    """
    创建关键帧筛选器实例
    
    Args:
        translation_threshold: 平移阈值（米）
        rotation_threshold: 旋转阈值（度）
        
    Returns:
        KeyframeFilter实例
    """
    return KeyframeFilter(translation_threshold, rotation_threshold)
