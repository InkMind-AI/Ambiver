"""Centralized path configuration for AmbiVer."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
    """Repository root (parent of src/)."""
    env = os.environ.get("AMBIVER_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent.parent


def bev_maps_dir() -> Path:
    env = os.environ.get("BEV_MAPS_DIR")
    if env:
        return Path(env)
    return project_root() / "bev_maps"


def grounding_dino_root() -> Path:
    return Path(os.environ.get("GROUNDING_DINO_ROOT", "GroundingDINO"))


def grounding_dino_config_path() -> Path:
    return grounding_dino_root() / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"


def grounding_dino_weights_path() -> Path:
    return grounding_dino_root() / "weights" / "groundingdino_swint_ogc.pth"


def groundingdino_python() -> Path:
    """Python executable for optional separate GroundingDINO conda env."""
    env = os.environ.get("GROUNDINGDINO_PYTHON")
    if env:
        return Path(env)
    conda_env = os.environ.get("GROUNDINGDINO_ENV")
    if conda_env:
        return Path(conda_env) / "bin" / "python"
    return Path(sys.executable)


def ensure_grounding_dino_on_path() -> Path:
    root = grounding_dino_root().resolve()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
