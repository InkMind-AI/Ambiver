"""
4.3 Reasoning Core: From Evidence to Verdict
============================================

Zero-shot LLM reasoning core that:
1. Packages structured evidence into LLM-friendly format
2. Uses VLM to evaluate instruction-instance conflicts
3. Generates ambiguity verdicts and clarification questions
"""
from __future__ import annotations
import json
import os
import time
from typing import List, Dict, Any, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass, asdict
from enum import Enum
import openai
import base64
import io
from pathlib import Path
import numpy as np
from PIL import Image
if TYPE_CHECKING:
    from .perception import ParsedInstruction, InstanceCandidate

def encode_image_to_base64(image: np.ndarray) -> str:
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    pil_image = Image.fromarray(image)
    buffered = io.BytesIO()
    pil_image.save(buffered, format='JPEG', quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'data:image/jpeg;base64,{img_str}'

class VerdictLabel(Enum):
    """Verdict labels for ambiguity detection"""
    UNAMBIGUOUS = 'Unambiguous'
    AMBIGUOUS = 'Ambiguous'

class ConflictType(Enum):
    """Types of instruction-instance conflicts"""
    INSTANCE = 'Instance'
    ATTRIBUTE = 'Attribute'
    SPATIAL = 'Spatial'
    ACTION = 'Action'

@dataclass
class Verdict:
    """Final verdict from the reasoning core"""
    label: VerdictLabel
    types: List[ConflictType]
    sources: List[str]
    explanation: str
    clarify: Optional[str] = None

@dataclass
class Dossier:
    """Structured evidence package for LLM reasoning"""
    instruction: Dict[str, Any]
    instances: List[Dict[str, Any]]
    meta: Dict[str, Any]

class EvidenceBundler:
    """4.3.1 Structured Evidence Bundling"""

    def __init__(self):
        """Initialize evidence bundler"""
        pass

    def create_dossier(self, parsed_instruction: 'ParsedInstruction', candidates: List['InstanceCandidate'], raw_instruction: str) -> Dossier:
        """
        Create structured evidence dossier for LLM reasoning
        
        Args:
            parsed_instruction: Parsed instruction components
            candidates: Unified instance candidates
            raw_instruction: Original natural language instruction
            
        Returns:
            Structured dossier for LLM processing
        """
        instances = []
        for candidate in candidates:
            instance_data = {'id': candidate.id, 'image': candidate.representative_image, 'bbox': candidate.representative_bbox, 'score': candidate.score, 'detection_count': len(candidate.detections)}
            instances.append(instance_data)
        instruction_data = {'raw': raw_instruction, 'parsed': {'Target': parsed_instruction.target, 'Attributes': parsed_instruction.attributes, 'Relations': parsed_instruction.relations, 'Action': parsed_instruction.action}}
        meta_data = {'camera_known': True, 'notes': 'training-free perception; thresholds are fixed constants', 'total_instances': len(candidates), 'processing_method': 'geometric_consistency'}
        return Dossier(instruction=instruction_data, instances=instances, meta=meta_data)

class LLMReasoner:
    """4.3.2 LLM as Zero-Shot Logical Adjudicator"""

    def __init__(self, model_name: str='qwen3-vl-30b-a3b-instruct', api_key: Optional[str]=None, base_url: Optional[str]=None):
        """
        Initialize LLM reasoner
        
        Args:
            model_name: LLM model to use (qwen3-vl-30b-a3b-instruct, qwen-vl-max, etc.)
            api_key: DashScope API key
            base_url: Custom API base URL
        """
        self.model_name = model_name
        resolved_api_key = api_key or os.getenv('DASHSCOPE_API_KEY') or os.getenv('OPENAI_API_KEY')
        if not resolved_api_key:
            raise ValueError('Missing API key: set DASHSCOPE_API_KEY or OPENAI_API_KEY, or pass api_key to LLMReasoner.')
        original_proxy = {}
        for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
            if key in os.environ:
                original_proxy[key] = os.environ[key]
                del os.environ[key]
        try:
            self.client = openai.OpenAI(api_key=resolved_api_key, base_url=base_url or 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        finally:
            for key, value in original_proxy.items():
                os.environ[key] = value
        self.last_messages = None
        self.last_raw_response = None

    def _create_reasoning_prompt(self, dossier: Dossier) -> str:
        """
        Create reasoning prompt for the LLM
        
        Args:
            dossier: Structured evidence package
            
        Returns:
            Formatted prompt string
        """
        prompt = f'\nYou are an expert robotic instruction analyzer. Your task is to evaluate whether a natural language instruction is ambiguous given the detected object instances in the scene.\n\nINSTRUCTION TO ANALYZE:\nRaw: "{dossier.instruction['raw']}"\nTarget: {dossier.instruction['parsed']['Target']}\nAttributes: {dossier.instruction['parsed']['Attributes']}\nRelations: {dossier.instruction['parsed']['Relations']}\nAction: {dossier.instruction['parsed']['Action']}\n\nDETECTED INSTANCES:\n'
        for instance in dossier.instances:
            prompt += f'\n- Instance {instance['id']}: \n  Image: {instance['image']}\n  Bounding box: {instance['bbox']}\n  Confidence: {instance['score']:.3f}\n  Detection count: {instance['detection_count']}\n'
        prompt += f'\n\nANALYSIS TASK:\nAmbiguity is execution-oriented: label Ambiguous only when missing information or vague wording would force risky guesswork or require clarification for safe completion. Acceptable vagueness without conflicting actions (e.g., "clean the table/room") should be labeled Unambiguous.\n\nDecision rubric (use only these types: Instance, Attribute, Spatial, Action):\n- Instance (referential): multiple objects of the target class; identity cannot be uniquely isolated.\n- Attribute (referential): subjective/relative attributes without uniqueness (e.g., "nice", "large") cause multiple valid matches.\n- Spatial (referential): viewpoint-dependent/underspecified relations (e.g., "left of", "near") yield multiple valid targets.\n- Action (execution): target is unique, but the verb implies mutually exclusive actions (e.g., upright/move/discard) requiring clarification.\n\nProcedure:\n1) If any of Instance/Attribute/Spatial applies, include those types.\n2) If the verb has mutually exclusive actions requiring clarification, include Action.\n3) Otherwise label Unambiguous.\n\nEvidence & uncertainty:\n- Prefer citing BEV/view evidence when available.\n- Action ambiguity may be justified linguistically (even without visuals) if it implies concrete conflicting actions.\n- For referential subtypes, use scene cues when present; if coverage is clearly limited and uniqueness cannot be established, state why clarification is needed.\n\nMinimal examples:\n- "clean the table" → Unambiguous\n- "pick up the cup" with multiple cups and no unique qualifier → Ambiguous (Instance)\n- "deal with the cup" (upright/move/discard) → Ambiguous (Action)\n\nRESPONSE FORMAT (JSON). Choose label: "Ambiguous" or "Unambiguous".\n{'label': \"Ambiguous\",\n    \"types\": [\"Instance\", \"Attribute\", \"Spatial\", \"Action\"],\n    \"explanation\": \"Brief analysis citing BEV/local evidence or linguistic grounds\",\n    \"clarify\": \"Optional concise clarification question\"\n}\n\nProvide only the JSON response, no additional text.\n'
        return prompt

    def _extract_first_json(self, s: str) -> Optional[str]:
        s = s.strip()
        if s.startswith('```json'):
            s = s[7:]
        if s.startswith('```'):
            s = s[3:]
        s = s.strip()
        start = s.find('{')
        if start == -1:
            return None
        depth = 0
        for i, c in enumerate(s[start:], start):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
        return None

    def _extract_last_valid_json(self, s: str) -> Optional[str]:
        s = s.strip()
        if s.startswith('```json'):
            s = s[7:]
        if s.startswith('```'):
            s = s[3:]
        s = s.strip()
        candidates = []
        pos = 0
        while True:
            start = s.find('{', pos)
            if start == -1:
                break
            depth = 0
            for i, c in enumerate(s[start:], start):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        candidates.append((start, i + 1))
                        pos = i + 1
                        break
            else:
                break
        for start, end in reversed(candidates):
            json_str = s[start:end]
            try:
                data = json.loads(json_str)
                if 'label' in data and data['label'] in ('Ambiguous', 'Unambiguous'):
                    return json_str
            except json.JSONDecodeError:
                continue
        return self._extract_first_json(s) if candidates else None

    def _parse_verdict_response(self, response: str) -> Verdict:
        """
        Parse LLM response into Verdict object
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed verdict object
        """
        try:
            json_str = self._extract_last_valid_json(response)
            if not json_str:
                raise ValueError('No JSON object found in response')
            data = json.loads(json_str)
            label = VerdictLabel.UNAMBIGUOUS if data.get('label') == 'Unambiguous' else VerdictLabel.AMBIGUOUS
            types = []
            for type_str in data.get('types', []):
                try:
                    types.append(ConflictType(type_str))
                except ValueError:
                    continue
            sources = data.get('sources', [])
            explanation = data.get('explanation', '')
            clarify = data.get('clarify')
            return Verdict(label=label, types=types, sources=sources, explanation=explanation, clarify=clarify)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return Verdict(label=VerdictLabel.AMBIGUOUS, types=[ConflictType.INSTANCE], sources=[], explanation=f'Failed to parse LLM response: {str(e)}', clarify='Could you please rephrase your instruction more clearly?')

    def reason(self, dossier: Dossier, instance_images: Dict[str, str], birdseye_image_b64: Optional[str]=None) -> Verdict:
        """
        Perform zero-shot reasoning on the evidence dossier with visual input
        
        Args:
            dossier: Structured evidence package
            instance_images: Dict mapping instance IDs to base64-encoded images
            
        Returns:
            Verdict with ambiguity analysis
        """
        try:
            if birdseye_image_b64 is not None:
                scene_description = "We provide a bird's-eye map of the scene and a subset of view images.\nNote: Each view may cover only part of the scene; some areas may not be captured by cameras.\nMultiple images may correspond to the same object from different viewpoints - these overlapping scene details provide comprehensive visual context for analysis.\nUse the BEV to reason about global layout, and use the view images for local evidence."
            else:
                scene_description = 'We provide a subset of view images from the scene.\nNote: Each view may cover only part of the scene; some areas may not be captured by cameras.\nMultiple images may correspond to the same object from different viewpoints - these overlapping scene details provide comprehensive visual context for analysis.\nUse the view images for local evidence analysis.'
            attrs = dossier.instruction['parsed'].get('Attributes', [])
            attrs_str = ', '.join(attrs) if isinstance(attrs, list) else str(attrs)
            parsed_line = f'Target: {dossier.instruction['parsed']['Target']};Attributes: {attrs_str};Relations: {dossier.instruction['parsed']['Relations']}; Action: {dossier.instruction['parsed'].get('Action') or 'N/A'}'
            content_parts = [{'type': 'text', 'text': f'Instruction: {dossier.instruction['raw']}\n\nParsed Components:\n{parsed_line}\n\nVisual Information:\n{scene_description}'}]
            if birdseye_image_b64 is not None:
                content_parts.append({'type': 'text', 'text': "Bird's-eye view (top-down orthographic rendering) of the scene:"})
                content_parts.append({'type': 'image_url', 'image_url': {'url': birdseye_image_b64}})
            for instance in dossier.instances[:10]:
                instance_id = instance['id']
                bbox = instance.get('bbox', [])
                score = instance.get('score', 0.0)
                det_count = instance.get('detection_count', 0)
                inst_text = f'\nInstance {instance_id}:\n- Bounding box: {bbox}\n- Confidence score: {score:.3f}\n- Detection count: {det_count} views'
                if instance_id in instance_images:
                    content_parts.append({'type': 'image_url', 'image_url': {'url': instance_images[instance_id]}})
                    content_parts.append({'type': 'text', 'text': inst_text})
                else:
                    content_parts.append({'type': 'text', 'text': f'\nInstance {instance_id} (no image available):\n- Bounding box: {bbox}\n- Confidence score: {score:.3f}\n- Detection count: {det_count} views'})
            content_parts.append({'type': 'text', 'text': '\n\nTask: Analyze if this instruction is ambiguous based on the detected instances.\n\nAmbiguity is execution-oriented: label Ambiguous only when safe execution would require clarification or risky guesswork. Acceptable vagueness without conflicting actions (e.g., "clean the table/room") → Unambiguous.\n\nTypes (only use these):\nInstance: multiple objects plausibly match the target AND cannot be uniquely identified by spatial constraints or context\nAttribute: subjective/relative attributes without uniqueness\nSpatial: viewpoint-dependent/underspecified relations yield multiple targets (e.g., "behind", "left of", "near" depend on observer position)\nAction: verb implies mutually exclusive actions requiring clarification\n\nResponse format: { "label": "Ambiguous" or "Unambiguous", "types": ["Instance", "Attribute", "Spatial", "Action"], "explanation": "Brief explanation citing BEV/global vs local evidence or linguistic grounds", "clarify": "Optional clarifying question to resolve ambiguity" }'})
            messages = [{'role': 'system', 'content': 'You are an expert analyzing ambiguity in robotic instructions using an execution-oriented criterion: label Ambiguous only when safe execution would require clarification or risky guesswork. Only use types: Instance, Attribute, Spatial, Action.'}, {'role': 'user', 'content': content_parts}]
            self.last_messages = messages
            response = self.client.chat.completions.create(model=self.model_name, messages=messages, temperature=0.0, max_tokens=1000)
            response_text = response.choices[0].message.content
            self.last_raw_response = response_text
            return self._parse_verdict_response(response_text)
        except Exception as e:
            self.last_raw_response = None
            print(f'LLM reasoning error: {e}')
            return Verdict(label=VerdictLabel.AMBIGUOUS, types=[ConflictType.INSTANCE], sources=[], explanation=f'LLM reasoning failed: {str(e)}', clarify='Could you please rephrase your instruction more clearly?')

class LocalQwen3VLReasoner:

    def __init__(self, model_name: str='Qwen/Qwen3-VL-8B-Instruct', device_map: str='auto', lora_path: Optional[str]=None):
        import torch
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        self.model_name = model_name
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device_map)
        if lora_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model.eval()
        self.processor = AutoProcessor.from_pretrained(lora_path or model_name)
        self.last_messages = None
        self.last_raw_response = None

    def _b64_to_numpy(self, b64_url: str) -> np.ndarray:
        import base64
        from PIL import Image
        import io
        if b64_url.startswith('data:image'):
            b64 = b64_url.split(',', 1)[1]
        else:
            b64 = b64_url
        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        return np.array(img)

    def _build_paper_prompt_tail(self, dossier: Dossier) -> str:
        return '\n\nTask: Analyze if this instruction is ambiguous based on the detected instances.\n\nAmbiguity is execution-oriented: label Ambiguous only when safe execution would require clarification or risky guesswork. Acceptable vagueness without conflicting actions (e.g., "clean the table/room") → Unambiguous.\n\nTypes (only use these):\nInstance: multiple objects plausibly match the target AND cannot be uniquely identified by spatial constraints or context\nAttribute: subjective/relative attributes without uniqueness\nSpatial: viewpoint-dependent/underspecified relations yield multiple targets (e.g., "behind", "left of", "near" depend on observer position)\nAction: verb implies mutually exclusive actions requiring clarification\n\nResponse format: { "label": "Ambiguous" or "Unambiguous", "types": ["Instance", "Attribute", "Spatial", "Action"], "explanation": "Brief explanation citing BEV/global vs local evidence or linguistic grounds", "clarify": "Optional clarifying question to resolve ambiguity" }'

    def reason(self, dossier: Dossier, instance_images: Dict[str, str], birdseye_image_b64: Optional[str]=None) -> Verdict:
        api_reasoner = LLMReasoner(model_name='qwen-vl-plus', api_key=os.getenv('DASHSCOPE_API_KEY', 'dummy'), base_url='https://dashscope.aliyuncs.com/compatible-mode/v1')
        attrs = dossier.instruction['parsed'].get('Attributes', [])
        attrs_str = ', '.join(attrs) if isinstance(attrs, list) else str(attrs)
        parsed_line = f'Target: {dossier.instruction['parsed']['Target']};Attributes: {attrs_str};Relations: {dossier.instruction['parsed']['Relations']}; Action: {dossier.instruction['parsed'].get('Action') or 'N/A'}'
        scene_desc = "We provide a bird's-eye map of the scene and a subset of view images. Note: Each view may cover only part of the scene; some areas may not be captured by cameras. Multiple images may correspond to the same object from different viewpoints - these overlapping scene details provide comprehensive visual context for analysis. Use the BEV to reason about global layout, and use the view images for local evidence."
        if birdseye_image_b64 is None:
            scene_desc = 'We provide a subset of view images from the scene. Note: Each view may cover only part of the scene; some areas may not be captured by cameras. Multiple images may correspond to the same object from different viewpoints - these overlapping scene details provide comprehensive visual context for analysis. Use the view images for local evidence analysis.'
        content_parts = []
        content_parts.append({'type': 'text', 'text': f'Instruction: {dossier.instruction['raw']}\n\nParsed Components:\n{parsed_line}\n\nVisual Information:\n{scene_desc}'})
        images_list = []
        if birdseye_image_b64:
            content_parts.append({'type': 'text', 'text': "Bird's-eye view (top-down orthographic rendering) of the scene:\n\n"})
            content_parts.append({'type': 'image', 'image': self._b64_to_numpy(birdseye_image_b64)})
            images_list.append(self._b64_to_numpy(birdseye_image_b64))
        for instance in dossier.instances[:10]:
            iid = instance['id']
            bbox = instance.get('bbox', [])
            score = instance.get('score', 0.0)
            det_count = instance.get('detection_count', 0)
            if iid in instance_images:
                content_parts.append({'type': 'image', 'image': self._b64_to_numpy(instance_images[iid])})
                images_list.append(self._b64_to_numpy(instance_images[iid]))
                content_parts.append({'type': 'text', 'text': f'\nInstance {iid}:\n- Bounding box: {bbox}\n- Confidence score: {score:.3f}\n- Detection count: {det_count} views'})
            else:
                content_parts.append({'type': 'text', 'text': f'\nInstance {iid} (no image available):\n- Bounding box: {bbox}\n- Confidence score: {score:.3f}\n- Detection count: {det_count} views'})
        content_parts.append({'type': 'text', 'text': self._build_paper_prompt_tail(dossier)})
        if not images_list:
            raise ValueError('No visual evidence: Dossier requires real BEV (I_bev) per §3.3. Ensure birdseye_image is loaded before reasoning; no placeholder allowed.')
        messages = [{'role': 'user', 'content': content_parts}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images_list, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        import torch
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        response = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        if '{' in response and '}' in response:
            response = response[response.find('{'):response.rfind('}') + 1]
        model_part = api_reasoner._extract_last_valid_json(response)
        self.last_raw_response = model_part if model_part else response
        return api_reasoner._parse_verdict_response(response)

class ReasoningCore:
    """Main reasoning core combining evidence bundling and LLM reasoning"""

    def __init__(self, model_name: str='qwen3-vl-30b-a3b-instruct', api_key: Optional[str]=None, base_url: Optional[str]=None, use_local_model: bool=False, device_map: str='auto', **kwargs):
        """
        Initialize reasoning core
        
        Args:
            model_name: LLM model to use
            api_key: DashScope API key
            base_url: Custom API base URL
            use_local_model: Use local Qwen3-VL-8B instead of API
            device_map: Device map for local model
        """
        self.bundler = EvidenceBundler()
        if use_local_model:
            local_model = kwargs.get('local_model_name', 'Qwen/Qwen3-VL-8B-Instruct')
            lora_path = kwargs.get('lora_path')
            self.reasoner = LocalQwen3VLReasoner(model_name=local_model, device_map=device_map, lora_path=lora_path)
        else:
            self.reasoner = LLMReasoner(model_name, api_key, base_url)

    def process(self, parsed_instruction: 'ParsedInstruction', candidates: List['InstanceCandidate'], raw_instruction: str, images: List[np.ndarray], birdseye_image: Optional[np.ndarray]=None) -> Verdict:
        """
        Main reasoning pipeline: bundle evidence and generate verdict
        
        Args:
            parsed_instruction: Parsed instruction components
            candidates: Unified instance candidates
            raw_instruction: Original natural language instruction
            images: List of scene images (all views)
            
        Returns:
            Final verdict with ambiguity analysis
        """
        dossier = self.bundler.create_dossier(parsed_instruction, candidates, raw_instruction)
        self.last_dossier = dossier
        instance_images = {}
        sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        limited_candidates = sorted_candidates[:6]
        print(f'LLM instances: {len(limited_candidates)} (sorted by confidence)')
        for i, candidate in enumerate(limited_candidates):
            print(f'  {i + 1}. {candidate.id}: {candidate.representative_image} (confidence: {candidate.score:.4f})')
        for candidate in limited_candidates:
            try:
                view_id = int(candidate.representative_image.split('_')[1].split('.')[0])
                if view_id >= len(images):
                    print(f'Warning: view_id {view_id} >= len(images) {len(images)}, skipping image for {candidate.id}')
                    continue
                instance_images[candidate.id] = encode_image_to_base64(images[view_id])
            except (ValueError, IndexError) as e:
                print(f"Warning: Failed to parse view_id from '{candidate.representative_image}': {e}, skipping image for {candidate.id}")
                continue
        birdseye_b64 = None
        if birdseye_image is not None:
            try:
                birdseye_b64 = encode_image_to_base64(birdseye_image)
            except Exception:
                birdseye_b64 = None
        llm_start = time.time()
        verdict = self.reasoner.reason(dossier, instance_images, birdseye_b64)
        self.last_reasoning_time = time.time() - llm_start
        return verdict