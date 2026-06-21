"""
AmbiJudge: Main Interface
========================

Unified interface for the AmbiJudge system combining perception and reasoning.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import time

from .perception import PerceptionEngine, CameraParams, Detection, ParsedInstruction, InstanceCandidate
from .reasoning import ReasoningCore, Verdict


class AmbiJudge:
    """
    Main AmbiJudge interface combining perception and reasoning modules
    
    This class provides the complete pipeline from multi-view images and
    natural language instructions to ambiguity verdicts.
    """
    
    def __init__(self, 
                 perception_kwargs: Optional[Dict[str, Any]] = None,
                 reasoning_kwargs: Optional[Dict[str, Any]] = None,
                 # H-VIM parameters
                 use_hvim: bool = True, voxel_size: float = 0.05, K_max: int = 16,
                 tau_iou: float = 0.25, tau_iom: float = 0.60, min_detections: int = 2,
                 use_gpu: bool = True, max_memory_gb: float = 4.0,
                 scannet_root: Optional[str] = None):
        """
        Initialize AmbiJudge system
        
        Args:
            perception_kwargs: Parameters for perception engine
            reasoning_kwargs: Parameters for reasoning core
            use_hvim: Enable H-VIM GPU-accelerated merging
            voxel_size: Voxel size in meters for H-VIM
            K_max: Maximum detections per voxel (hot voxel truncation)
            tau_iou: IoU threshold for merging
            tau_iom: IoM threshold for merging
            min_detections: Minimum detections per instance
            use_gpu: Enable GPU acceleration for H-VIM
            max_memory_gb: Maximum GPU memory to use
        """
        # Merge H-VIM parameters with perception_kwargs
        if perception_kwargs is None:
            perception_kwargs = {}
        
        perception_kwargs.update({
            'use_hvim': use_hvim,
            'voxel_size': voxel_size,
            'K_max': K_max,
            'tau_iou': tau_iou,
            'tau_iom': tau_iom,
            'min_detections': min_detections,
            'use_gpu': use_gpu,
            'max_memory_gb': max_memory_gb,
            'scannet_root': scannet_root
        })
        
        self.perception_engine = PerceptionEngine(**(perception_kwargs or {}))
        self.reasoning_core = ReasoningCore(**(reasoning_kwargs or {}))
        
        # 用于存储最后一次处理的时间信息
        self.last_timing_info = {}
    
    def process(self, 
                instruction: str,
                images: List[np.ndarray],
                cameras: List[CameraParams],
                detections: List[Detection],
                birdseye_image: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Complete AmbiJudge pipeline processing
        
        Args:
            instruction: Natural language instruction
            images: List of input images (H, W, 3)
            cameras: Camera parameters for each view
            detections: Object detections from GroundingDINO
            
        Returns:
            Complete processing results including verdict
        """
        # 重置时间信息
        self.last_timing_info = {}
        
        # Step 1: Perception Engine (4.2) - 记录实例合并时间
        perception_start = time.time()
        parsed_instruction, candidates = self.perception_engine.process(
            instruction, images, cameras, detections
        )
        perception_time = time.time() - perception_start
        
        # 从perception_engine获取实例合并时间（如果已记录）
        if hasattr(self.perception_engine, 'last_unification_time'):
            self.last_timing_info["instance_unification"] = self.perception_engine.last_unification_time
        else:
            # 如果没有单独记录，使用总时间（通常实例合并很快）
            self.last_timing_info["instance_unification"] = perception_time
        
        # Step 2: Reasoning Core (4.3) - 记录LLM推理时间
        reasoning_start = time.time()
        verdict = self.reasoning_core.process(
            parsed_instruction, candidates, instruction, images, birdseye_image
        )
        reasoning_time = time.time() - reasoning_start
        
        # 从reasoning_core获取LLM推理时间（如果已记录）
        if hasattr(self.reasoning_core, 'last_reasoning_time'):
            self.last_timing_info["llm_reasoning"] = self.reasoning_core.last_reasoning_time
        else:
            self.last_timing_info["llm_reasoning"] = reasoning_time
        
        # Package results
        # expose exact LLM input (messages + dossier/meta) for test/debug saving
        llm_messages = getattr(self.reasoning_core.reasoner, 'last_messages', None)
        llm_raw_response = getattr(self.reasoning_core.reasoner, 'last_raw_response', None)
        last_dossier = getattr(self.reasoning_core, 'last_dossier', None)
        dossier_dict = None
        if last_dossier is not None:
            dossier_dict = {
                "instruction": last_dossier.instruction,
                "instances": last_dossier.instances,
                "meta": last_dossier.meta,
            }
        results = {
            "instruction": {
                "raw": instruction,
                "parsed": {
                    "target": parsed_instruction.target,
                    "attributes": parsed_instruction.attributes,
                    "relations": parsed_instruction.relations,
                    "action": parsed_instruction.action
                }
            },
            "candidates_found": [
                {
                    "id": candidate.id,
                    "representative_image": candidate.representative_image,
                    "representative_bbox": candidate.representative_bbox,
                    "score": candidate.score,
                    "detection_count": len(candidate.detections)
                }
                for candidate in candidates
            ],
            "count": len(candidates),
            "verdict": {
                "label": verdict.label.value,
                "types": [t.value for t in verdict.types],
                "sources": verdict.sources,
                "explanation": verdict.explanation,
                "clarify": verdict.clarify
            },
            "llm_input": {
                "messages": llm_messages,
                "dossier": dossier_dict,
            },
            "llm_raw_response": llm_raw_response,
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Save processing results to JSON file
        
        Args:
            results: Processing results from process()
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def load_results(self, input_path: str) -> Dict[str, Any]:
        """
        Load processing results from JSON file
        
        Args:
            input_path: Input file path
            
        Returns:
            Loaded processing results
        """
        with open(input_path, 'r') as f:
            return json.load(f)
