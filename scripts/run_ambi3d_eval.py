"""
Ambi3D Evaluation Runner for AmbiJudge
=====================================

This script evaluates AmbiJudge on the ambi3d.json dataset using ScanNet scenes.
It loads the dataset, runs the full AmbiJudge pipeline (perception + reasoning),
extracts binary predictions (0/1), and computes accuracy metrics against ground truth.

Usage:
    python run_ambi3d_eval.py --dataset ambi3d.json --scannet_root /path/to/scannet --output_dir results/
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
import cv2
import numpy as np
from tqdm import tqdm
import torch
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from ambijudge.ambijudge import AmbiJudge
from ambijudge.bev_utils import generate_bev_from_depth_scene
from ambijudge.paths import grounding_dino_root, project_root
from ambijudge.perception import CameraParams, InstructionParser
from ambijudge.subprocess_detector import SubprocessDetector
from ambijudge.keyframe_filter import KeyframeFilter
_HAS_BEV_UTILS = True
_OOM_EXCEPTIONS = (RuntimeError,)
if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'OutOfMemoryError'):
    _OOM_EXCEPTIONS = (RuntimeError, torch.cuda.OutOfMemoryError)

def _load_birdseye(scannet_root: str, scene_id: str) -> Optional[np.ndarray]:
    if not _HAS_BEV_UTILS:
        return None
    try:
        _, color_img = generate_bev_from_depth_scene(scannet_root, scene_id)
        if color_img is not None and color_img.ndim == 3:
            return cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        if color_img is not None:
            return np.asarray(color_img)
    except FileNotFoundError:
        pass
    return None

class ScanNetSceneLoader:

    def __init__(self, scannet_root: str):
        self.scannet_root = Path(scannet_root)
        if not self.scannet_root.exists():
            raise ValueError(f'ScanNet root not found: {scannet_root}')

    def load_scene_images(self, scene_id: str, max_frames: Optional[int]=None, use_all_frames: bool=False, use_keyframe_filter: bool=True, target_frames: Optional[int]=None) -> Tuple[List[np.ndarray], List[CameraParams]]:
        scene_path = self.scannet_root / scene_id
        if not scene_path.exists():
            raise ValueError(f'Scene not found: {scene_path}')
        color_dir = scene_path / 'color'
        if color_dir.exists():
            image_files = sorted([f for f in color_dir.glob('*.jpg')])
        else:
            image_files = sorted([f for f in scene_path.glob('*.jpg')])
        print(f'Found {len(image_files)} total images in scene {scene_id}')
        print('📷 Reading camera parameters...')
        all_cameras = []
        valid_indices = []
        for i, img_file in enumerate(image_files):
            if img_file.parent.name == 'color':
                pose_file = img_file.parent.parent / 'pose' / (img_file.stem + '.txt')
            else:
                pose_file = img_file.with_suffix('.txt')
            if pose_file.exists():
                camera_params = self._load_camera_params(pose_file)
                all_cameras.append(camera_params)
                valid_indices.append(i)
            else:
                print(f'Warning: Camera pose file not found: {pose_file}')
                continue
        print(f'Successfully read {len(all_cameras)} camera parameters')
        if use_keyframe_filter and len(all_cameras) > 1:
            print(f'🔍 Applying keyframe filtering to {len(all_cameras)} frames...')
            keyframe_filter = KeyframeFilter(translation_threshold=0.15, rotation_threshold=15.0)
            filtered_cameras, keyframe_indices = keyframe_filter._filter_by_poses(all_cameras, target_frames)
            adaptive_trigger = 1.5 * 25
            if len(filtered_cameras) > adaptive_trigger:
                print(f'Still {len(filtered_cameras)} frames after filtering, adapting thresholds...')
                target_frames_adaptive = 25
                best_result = (filtered_cameras, keyframe_indices)
                best_diff = abs(len(filtered_cameras) - target_frames_adaptive)
                for iteration in range(8):
                    if len(filtered_cameras) > target_frames_adaptive:
                        keyframe_filter.translation_threshold *= 1.5
                        keyframe_filter.rotation_threshold *= 1.5
                    else:
                        keyframe_filter.translation_threshold *= 0.7
                        keyframe_filter.rotation_threshold *= 0.7
                    keyframe_filter.translation_threshold = max(0.05, min(2.0, keyframe_filter.translation_threshold))
                    keyframe_filter.rotation_threshold = max(5.0, min(90.0, keyframe_filter.rotation_threshold))
                    print(f'  Iteration {iteration + 1}: thresholds=({keyframe_filter.translation_threshold:.3f}m, {keyframe_filter.rotation_threshold:.1f} deg), target={target_frames_adaptive}')
                    filtered_cameras_iter, filtered_indices_iter = keyframe_filter._filter_by_poses(all_cameras, target_frames)
                    current_diff = abs(len(filtered_cameras_iter) - target_frames_adaptive)
                    if current_diff < best_diff:
                        best_result = (filtered_cameras_iter, filtered_indices_iter)
                        best_diff = current_diff
                    print(f'    Result: {len(filtered_cameras_iter)} frames (diff: {current_diff})')
                    if current_diff <= 5:
                        break
                filtered_cameras, keyframe_indices = best_result
                print(f'Adaptive tuning done: final thresholds=({keyframe_filter.translation_threshold:.3f}m, {keyframe_filter.rotation_threshold:.1f} deg)')
            print(f'Keyframe filtering: {len(all_cameras)} -> {len(filtered_cameras)} frames')
            print(f'Reduction: {len(filtered_cameras) / len(all_cameras) * 100:.1f}%')
            print(f'Retained frame indices: {keyframe_indices[:10]}{('...' if len(keyframe_indices) > 10 else '')}')
            selected_files = [image_files[valid_indices[idx]] for idx in keyframe_indices]
            cameras = filtered_cameras
        elif use_all_frames:
            print(f'Using all {len(image_files)} frames (use_all_frames=True)')
            selected_files = image_files
            cameras = all_cameras
        elif max_frames is not None and len(image_files) > max_frames:
            print(f'Scene has {len(image_files)} frames, limiting to {max_frames} using uniform sampling')
            step = len(image_files) / max_frames
            selected_indices = [int(i * step) for i in range(max_frames)]
            selected_files = [image_files[i] for i in selected_indices]
            cameras = [all_cameras[i] for i in selected_indices]
            print(f'Sampling step: {step:.2f}, selected indices: {selected_indices[:5]}...{selected_indices[-5:]}')
        else:
            print(f'Scene has {len(image_files)} frames, using all frames')
            selected_files = image_files
            cameras = all_cameras
        print(f'🖼️  Loading {len(selected_files)} images...')
        images = []
        for i, img_file in enumerate(selected_files):
            img_bgr = cv2.imread(str(img_file))
            if img_bgr is None:
                raise ValueError(f'Failed to load image: {img_file}')
            images.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            if (i + 1) % 10 == 0 or i == len(selected_files) - 1:
                print(f'Loaded {i + 1}/{len(selected_files)} images')
        print(f'✅ Successfully loaded {len(images)} images and {len(cameras)} camera parameters')
        return (images, cameras)

    def _generate_camera_params(self, num_views: int) -> List[CameraParams]:
        cameras = []
        fx = fy = 577.5
        cx = cy = 320.0
        for i in range(num_views):
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            angle = i * 2 * math.pi / num_views
            radius = 2.0
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            y = 1.5
            target = np.array([0, 0, 0])
            position = np.array([x, y, z])
            forward = target - position
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, np.array([0, 1, 0]))
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            R = np.column_stack([right, up, -forward])
            t = -R @ position
            cameras.append(CameraParams(K=K, R=R, t=t))
        return cameras

    def _load_camera_params(self, camera_file: Path) -> CameraParams:
        transform_matrix = np.loadtxt(camera_file)
        R = transform_matrix[:3, :3]
        t = transform_matrix[:3, 3]
        fx = fy = 577.5
        cx = cy = 320.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return CameraParams(K=K, R=R, t=t)

    def _get_default_camera_params(self) -> CameraParams:
        fx = fy = 577.5
        cx = cy = 320.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        R = np.eye(3)
        t = np.zeros(3)
        return CameraParams(K=K, R=R, t=t)

class Ambi3DEvaluator:

    def __init__(self, scannet_root: str, verbose: bool=False, use_all_frames: bool=False, use_keyframe_filter: bool=True, target_frames: int=100, use_hvim: bool=True, voxel_size: float=0.05, K_max: int=16, tau_iou: float=0.25, tau_iom: float=0.6, min_detections: int=2, use_gpu: bool=True, max_memory_gb: float=4.0, multi_gpu: bool=False, gpu_ids: List[int]=[2, 3, 4, 5], chunk_size: int=8, batch_sizes: List[int]=[4, 6, 4], use_local_qwen3vl: bool=True, device_map: str='auto', model_name: str='Qwen/Qwen3-VL-8B-Instruct', use_batch_detector: bool=False):
        self.scannet_root = Path(scannet_root)
        self.scene_loader = ScanNetSceneLoader(scannet_root)
        self.verbose = verbose
        self.use_all_frames = use_all_frames
        self.use_keyframe_filter = use_keyframe_filter
        self.target_frames = target_frames
        self.use_hvim = use_hvim
        self.voxel_size = voxel_size
        self.K_max = K_max
        self.tau_iou = tau_iou
        self.tau_iom = tau_iom
        self.min_detections = min_detections
        self.use_gpu = use_gpu
        self.max_memory_gb = max_memory_gb
        self.multi_gpu = False
        self.gpu_ids = gpu_ids
        self.chunk_size = chunk_size
        self.batch_sizes = batch_sizes
        self.use_local_qwen3vl = use_local_qwen3vl
        self.device_map = device_map
        self.model_name = model_name
        reasoning_kwargs = self._get_reasoning_kwargs()
        self.ambijudge = AmbiJudge(reasoning_kwargs=reasoning_kwargs, use_hvim=use_hvim, voxel_size=voxel_size, K_max=K_max, tau_iou=tau_iou, tau_iom=tau_iom, min_detections=min_detections, use_gpu=use_gpu, max_memory_gb=max_memory_gb, scannet_root=scannet_root)
        if use_batch_detector:
            from ambijudge.batch_groundingdino_detector import BatchGroundingDINODetector
            self.detector = BatchGroundingDINODetector(batch_size=16)
            print('Using BatchGroundingDINODetector (batch=16, faster)')
        elif use_local_qwen3vl:
            self.detector = SubprocessDetector(gpu_id=0)
            print('Using SubprocessDetector for GroundingDINO')
        else:
            from ambijudge.batch_groundingdino_detector import BatchGroundingDINODetector
            self.detector = BatchGroundingDINODetector(batch_size=16)
            print('Using BatchGroundingDINODetector (batch=16)')
        self.results = []
        self.failed_questions = []
        self.confusion_matrix = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        self.ambiguity_type_stats = {}
        self._scene_cache = {}

    def _get_reasoning_kwargs(self) -> Dict[str, Any]:
        if not self.use_local_qwen3vl:
            raise ValueError('AmbiVer release defaults to local Qwen3-VL-8B. Pass --use_local_qwen3vl (default) or set use_local_qwen3vl=True.')
        return {'use_local_model': True, 'device_map': self.device_map, 'local_model_name': self.model_name}

    def evaluate_dataset(self, questions: List[Dict[str, Any]], output_dir: str, save_visualizations: bool=True, verbose: bool=True, resume: bool=False, checkpoint_every: int=1000) -> Dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        completed = set()
        if resume:
            checkpoint_file = output_path / 'detailed_results.json'
            if checkpoint_file.exists():
                try:
                    with open(checkpoint_file) as f:
                        self.results = json.load(f)
                    self._rebuild_stats_from_results()
                    completed = {(r['question']['scene_id'], r['question']['instruction_id']) for r in self.results}
                    print(f'Resumed: loaded {len(self.results)} existing results, {len(completed)} completed')
                except Exception as e:
                    print(f'Warning: Failed to load checkpoint: {e}, starting fresh')
                    self.results = []
            else:
                self.results = []
        scene_groups = self._group_questions_by_scene(questions)
        for scene_id, scene_questions in tqdm(scene_groups.items(), desc='Processing scenes'):
            pending = [q for q in scene_questions if (scene_id, q['instruction_id']) not in completed]
            if not pending:
                continue
            if verbose:
                print(f'\nProcessing scene: {scene_id} ({len(pending)} pending)')
            try:
                if scene_id in self._scene_cache:
                    if verbose:
                        print(f'Using cached scene data for {scene_id}')
                    images, cameras = self._scene_cache[scene_id]
                else:
                    images, cameras = self.scene_loader.load_scene_images(scene_id, use_all_frames=self.use_all_frames, use_keyframe_filter=self.use_keyframe_filter, target_frames=self.target_frames)
                    self._scene_cache[scene_id] = (images, cameras)
                birdseye_image = _load_birdseye(str(self.scannet_root), scene_id)
                if birdseye_image is None:
                    print(f'Skipping scene {scene_id}: no BEV (required by method, no placeholder)')
                    continue
                for question in pending:
                    try:
                        result = self._process_question(question, images, cameras, output_path, save_visualizations, output_dir, birdseye_image)
                        self.results.append(result)
                        completed.add((scene_id, question['instruction_id']))
                        if verbose:
                            self._print_question_result(question, result)
                    except _OOM_EXCEPTIONS as e:
                        if isinstance(e, RuntimeError) and 'out of memory' not in str(e).lower() and ('cuda' not in str(e).lower()):
                            raise
                        self._handle_oom_or_recoverable_error(question, scene_id, e, completed)
                        continue
                    except subprocess.CalledProcessError as e:
                        self._handle_oom_or_recoverable_error(question, scene_id, e, completed)
                        continue
                    except subprocess.TimeoutExpired as e:
                        self._handle_oom_or_recoverable_error(question, scene_id, e, completed)
                        continue
                    if len(self.results) % checkpoint_every == 0:
                        metrics = self._compute_metrics()
                        self._save_results(output_path, metrics)
                        print(f'Checkpoint: saved {len(self.results)} results')
            except Exception as e:
                import traceback
                print(f'Error processing scene {scene_id}: {e}')
                traceback.print_exc()
                continue
        metrics = self._compute_metrics()
        self._save_results(output_path, metrics)
        return metrics

    def _rebuild_stats_from_results(self):
        self.confusion_matrix = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        self.ambiguity_type_stats = {}
        for r in self.results:
            pred = r['predicted_answer']
            gt = r['ground_truth']
            if pred == 1 and gt == 1:
                self.confusion_matrix['TP'] += 1
            elif pred == 0 and gt == 0:
                self.confusion_matrix['TN'] += 1
            elif pred == 1 and gt == 0:
                self.confusion_matrix['FP'] += 1
            else:
                self.confusion_matrix['FN'] += 1
            amb_type = r['question'].get('ambiguity_type', 'Unknown')
            if amb_type not in self.ambiguity_type_stats:
                self.ambiguity_type_stats[amb_type] = {'correct': 0, 'total': 0}
            self.ambiguity_type_stats[amb_type]['total'] += 1
            if pred == gt:
                self.ambiguity_type_stats[amb_type]['correct'] += 1

    def _group_questions_by_scene(self, questions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        groups = {}
        for q in questions:
            scene_id = q['scene_id']
            if scene_id not in groups:
                groups[scene_id] = []
            groups[scene_id].append(q)
        return groups

    def _handle_oom_or_recoverable_error(self, question: Dict[str, Any], scene_id: str, exc: Exception, completed: set):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.failed_questions.append({'scene_id': scene_id, 'instruction_id': question.get('instruction_id'), 'question': question.get('question', '')[:100], 'error': str(exc)[:200]})
        completed.add((scene_id, question['instruction_id']))
        print(f"[OOM/Recoverable] Skipping {scene_id}/{question.get('instruction_id')}: {exc}")

    def _process_question(self, question: Dict[str, Any], images: List[np.ndarray], cameras: List[CameraParams], output_path: Path, save_visualizations: bool, output_dir: str, birdseye_image: np.ndarray) -> Dict[str, Any]:
        instruction_parser = InstructionParser()
        parsed_instruction = instruction_parser.parse(question['question'])
        target_text = parsed_instruction.target
        print(f"Extracted target noun: '{target_text}' from instruction: '{question['question']}'")
        print(f'Parsed instruction details:')
        print(f'  - Target: {parsed_instruction.target}')
        print(f'  - Attributes: {parsed_instruction.attributes}')
        print(f'  - Relations: {parsed_instruction.relations}')
        print(f'  - Action: {parsed_instruction.action}')
        if self.verbose:
            print(f'Running object detection for {len(images)} views...')
            print(f"GroundingDINO will search for: '{target_text}'")
        scene_id = question['scene_id']
        instruction_id = question['instruction_id']
        detections = self.detector.detect(images, target_text, visualization_dir=None)
        scene_path = str(self.scannet_root / scene_id)
        for det in detections:
            if not hasattr(det, 'metadata'):
                det.metadata = {}
            det.metadata['scene_path'] = scene_path
        print(f'Detection results:')
        print(f'  - Total detections: {len(detections)}')
        detections_by_view = {}
        for det in detections:
            view_id = det.view_id
            if view_id not in detections_by_view:
                detections_by_view[view_id] = 0
            detections_by_view[view_id] += 1
        print(f'  - Detections per view: {dict(sorted(detections_by_view.items()))}')
        if detections:
            avg_confidence = sum((det.confidence for det in detections)) / len(detections)
            print(f'  - Average confidence: {avg_confidence:.4f}')
            print(f'  - Confidence range: {min((det.confidence for det in detections)):.4f} - {max((det.confidence for det in detections)):.4f}')
        results = self.ambijudge.process(instruction=question['question'], images=images, cameras=cameras, detections=detections, birdseye_image=birdseye_image)
        candidates = results['candidates_found']
        print(f'Instance unification results:')
        print(f'  - Original detections: {len(detections)}')
        print(f'  - Unified instances: {len(candidates)}')
        ratio_str = f'{len(candidates) / len(detections) * 100:.1f}%' if len(detections) > 0 else 'N/A'
        print(f'  - Reduction ratio: {len(detections)} -> {len(candidates)} ({ratio_str})')
        for i, candidate in enumerate(candidates):
            print(f'  - Instance {candidate['id']}: {candidate['detection_count']} detections, score={candidate['score']:.4f}, rep_view={candidate['representative_image']}')
        predicted_answer = self._extract_binary_prediction(results['verdict'])
        ground_truth = question['answer']
        if predicted_answer == 1 and ground_truth == 1:
            self.confusion_matrix['TP'] += 1
        elif predicted_answer == 0 and ground_truth == 0:
            self.confusion_matrix['TN'] += 1
        elif predicted_answer == 1 and ground_truth == 0:
            self.confusion_matrix['FP'] += 1
        else:
            self.confusion_matrix['FN'] += 1
        ambiguity_type = question.get('ambiguity_type', 'Unknown')
        if ambiguity_type not in self.ambiguity_type_stats:
            self.ambiguity_type_stats[ambiguity_type] = {'correct': 0, 'total': 0}
        self.ambiguity_type_stats[ambiguity_type]['total'] += 1
        if predicted_answer == ground_truth:
            self.ambiguity_type_stats[ambiguity_type]['correct'] += 1
        slim_results = {'verdict': results['verdict'], 'llm_raw_response': results.get('llm_raw_response')}
        return {'question': question, 'results': slim_results, 'predicted_answer': predicted_answer, 'ground_truth': ground_truth, 'correct': predicted_answer == ground_truth}

    def _extract_binary_prediction(self, verdict: Dict[str, Any]) -> int:
        return 1 if verdict['label'] == 'Ambiguous' else 0

    def _print_question_result(self, question: Dict[str, Any], result: Dict[str, Any]):
        print(f'  Q: {question['question']}')
        print(f'  GT: {question['answer']} | Pred: {result['predicted_answer']} | Correct: {result['correct']}')
        print(f'  Verdict: {result['results']['verdict']['label']}')
        if result['results']['verdict']['types']:
            print(f'  Types: {', '.join(result['results']['verdict']['types'])}')

    def _compute_metrics(self) -> Dict[str, Any]:
        total = len(self.results)
        correct = sum((1 for r in self.results if r['correct']))
        accuracy = correct / total if total > 0 else 0.0
        tp = self.confusion_matrix['TP']
        tn = self.confusion_matrix['TN']
        fp = self.confusion_matrix['FP']
        fn = self.confusion_matrix['FN']
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        type_accuracies = {}
        for amb_type, stats in self.ambiguity_type_stats.items():
            type_accuracies[amb_type] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        return {'total_questions': total, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1, 'confusion_matrix': self.confusion_matrix, 'ambiguity_type_accuracies': type_accuracies, 'ambiguity_type_stats': self.ambiguity_type_stats}

    def _save_results(self, output_path: Path, metrics: Dict[str, Any]):
        with open(output_path / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        with open(output_path / 'detailed_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        predictions_data = []
        for result in self.results:
            predictions_data.append({'scene_id': result['question']['scene_id'], 'instruction_id': result['question']['instruction_id'], 'question': result['question']['question'], 'ground_truth': result['ground_truth'], 'predicted': result['predicted_answer'], 'correct': result['correct'], 'verdict_label': result['results']['verdict']['label'], 'llm_raw_response': result['results'].get('llm_raw_response')})
        with open(output_path / 'predictions.json', 'w') as f:
            json.dump(predictions_data, f, indent=2)
        failed = getattr(self, 'failed_questions', [])
        if failed:
            with open(output_path / 'failed_questions.json', 'w') as f:
                json.dump(failed, f, indent=2, ensure_ascii=False)
        print(f'\n{'=' * 50}')
        print('AMBI3D EVALUATION RESULTS')
        print(f'{'=' * 50}')
        print(f'Total Questions: {metrics['total_questions']}')
        print(f'Accuracy: {metrics['accuracy']:.3f}')
        print(f'Precision: {metrics['precision']:.3f}')
        print(f'Recall: {metrics['recall']:.3f}')
        print(f'F1 Score: {metrics['f1_score']:.3f}')
        print(f'\nConfusion Matrix:')
        print(f'  TP: {metrics['confusion_matrix']['TP']}')
        print(f'  TN: {metrics['confusion_matrix']['TN']}')
        print(f'  FP: {metrics['confusion_matrix']['FP']}')
        print(f'  FN: {metrics['confusion_matrix']['FN']}')
        print(f'\nAmbiguity Type Accuracies:')
        for amb_type, acc in metrics['ambiguity_type_accuracies'].items():
            stats = metrics['ambiguity_type_stats'][amb_type]
            print(f'  {amb_type}: {acc:.3f} ({stats['correct']}/{stats['total']})')
        failed = getattr(self, 'failed_questions', [])
        if failed:
            print(f"\nSkipped questions (OOM/error): {len(failed)} -> failed_questions.json")

def main():
    parser = argparse.ArgumentParser(description='Ambi3D Evaluation for AmbiJudge')
    parser.add_argument('--dataset', required=True, help='Path to ambi3d.json dataset')
    parser.add_argument('--scannet_root', required=True, help='Path to ScanNet dataset root')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--no_vis', action='store_true', help='Skip saving visualizations')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--use_all_frames', action='store_true', help='Use all available frames instead of limiting to 48')
    parser.add_argument('--use_keyframe_filter', action='store_true', default=True, help='Use keyframe filtering to reduce redundant frames')
    parser.add_argument('--target_frames', type=int, default=20, help='Target number of frames after keyframe filtering')
    parser.add_argument('--use_hvim', action='store_true', default=True, help='Use H-VIM GPU-accelerated instance merging')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='Voxel size in meters for H-VIM')
    parser.add_argument('--hvim_K_max', type=int, default=16, help='Hot voxel truncation threshold for H-VIM')
    parser.add_argument('--tau_iou', type=float, default=0.25, help='IoU threshold for H-VIM merging')
    parser.add_argument('--tau_iom', type=float, default=0.6, help='IoM threshold for H-VIM merging')
    parser.add_argument('--min_detections', type=int, default=2, help='Minimum detections per instance for H-VIM')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='Enable GPU acceleration for H-VIM')
    parser.add_argument('--max_memory_gb', type=float, default=4.0, help='Maximum GPU memory to use for H-VIM')
    parser.add_argument('--multi_gpu', action='store_true', help='Enable multi-GPU parallel inference')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[4, 5], help='GPU IDs to use (e.g. 4 5)')
    parser.add_argument('--use_local_qwen3vl', action='store_true', default=True, help='Use local Qwen3-VL-8B (default, training-free)')
    parser.add_argument('--model_name', default='Qwen/Qwen3-VL-8B-Instruct', help='HuggingFace model id for local VLM reasoning')
    parser.add_argument('--use_batch_detector', action='store_true', help='Run GroundingDINO in-process (same env as Qwen3-VL)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of questions (for testing)')
    parser.add_argument('--worker_id', type=int, default=None, help='Worker ID for parallel execution (0 to num_workers-1)')
    parser.add_argument('--num_workers', type=int, default=None, help='Total number of parallel workers')
    parser.add_argument('--resume', action='store_true', help='Resume from existing checkpoint')
    parser.add_argument('--checkpoint_every', type=int, default=1000, help='Save checkpoint every N questions')
    parser.add_argument('--chunk_size', type=int, default=8, help='Chunk size for multi-GPU processing')
    parser.add_argument('--batch_sizes', type=int, nargs=3, default=[4, 6, 4], help='Batch sizes for each shape bucket [800, 1024, 1280]')
    args = parser.parse_args()
    gd_root = grounding_dino_root()
    if not gd_root.exists():
        print(f'Warning: GroundingDINO not found at {gd_root}')
        print('Clone https://github.com/IDEA-Research/GroundingDINO and download weights.')
        print('Then set: export GROUNDING_DINO_ROOT=/path/to/GroundingDINO')
    bev_dir = project_root() / 'bev_maps'
    if not bev_dir.is_dir() or not any(bev_dir.glob('*.jpg')):
        print(f'Warning: BEV maps not found under {bev_dir}')
        print('Generate maps first: docs/BEV.md  |  python scripts/generate_bev_maps.py --help')
    with open(args.dataset, 'r') as f:
        questions = json.load(f)
    if args.limit:
        questions = questions[:args.limit]
        print(f'Limited to {len(questions)} questions')
    if args.worker_id is not None and args.num_workers is not None:
        questions = [q for i, q in enumerate(questions) if i % args.num_workers == args.worker_id]
        args.output_dir = os.path.join(args.output_dir, f'worker_{args.worker_id}')
        print(f'Worker {args.worker_id}/{args.num_workers}: processing {len(questions)} questions -> {args.output_dir}')
    print(f'Loaded {len(questions)} questions from {args.dataset}')
    device_map = 'auto'
    evaluator = Ambi3DEvaluator(args.scannet_root, verbose=args.verbose, use_all_frames=args.use_all_frames, use_keyframe_filter=args.use_keyframe_filter, target_frames=args.target_frames, use_hvim=args.use_hvim, voxel_size=args.voxel_size, K_max=args.hvim_K_max, tau_iou=args.tau_iou, tau_iom=args.tau_iom, min_detections=args.min_detections, use_gpu=args.use_gpu, max_memory_gb=args.max_memory_gb, multi_gpu=args.multi_gpu, gpu_ids=args.gpu_ids, chunk_size=args.chunk_size, batch_sizes=args.batch_sizes, use_local_qwen3vl=args.use_local_qwen3vl, device_map=device_map, model_name=args.model_name, use_batch_detector=args.use_batch_detector)
    metrics = evaluator.evaluate_dataset(questions=questions, output_dir=args.output_dir, save_visualizations=not args.no_vis, verbose=args.verbose, resume=args.resume, checkpoint_every=args.checkpoint_every)
    print(f'\nResults saved to: {args.output_dir}')
if __name__ == '__main__':
    main()