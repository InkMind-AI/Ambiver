#!/usr/bin/env python3
"""Generate top-down BEV images from ScanNet RGB-D (paper §3.2 / Appendix B.1).

Fuses multi-view depth into a point cloud and orthographically projects onto the
ground plane. For the exact paper pipeline, use BundleFusion; see docs/BEV.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_intrinsic(scene_path: Path) -> np.ndarray:
    for candidate in (
        scene_path / "intrinsic" / "intrinsic_color.txt",
        scene_path / "intrinsic" / "intrinsic_depth.txt",
    ):
        if candidate.exists():
            rows = []
            with open(candidate) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    rows.append([float(x) for x in line.split()])
            if len(rows) >= 3:
                return np.array(rows[:3], dtype=np.float64)
    return np.array([[577.870605, 0.0, 319.5], [0.0, 577.870605, 239.5], [0.0, 0.0, 1.0]])


def load_pose(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64).reshape(4, 4)


def backproject_frame(
    color_bgr: np.ndarray,
    depth_raw: np.ndarray,
    K: np.ndarray,
    pose: np.ndarray,
    depth_scale: float = 1000.0,
    stride: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth_raw.shape[:2]
    if color_bgr.shape[:2] != (h, w):
        color_bgr = cv2.resize(color_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    z = depth_raw.astype(np.float64) / depth_scale
    us = np.arange(0, w, stride)
    vs = np.arange(0, h, stride)
    uu, vv = np.meshgrid(us, vs)
    zz = z[vv, uu]
    valid = zz > 0.1
    if not np.any(valid):
        return np.empty((0, 3)), np.empty((0, 3))
    uu, vv, zz = uu[valid], vv[valid], zz[valid]
    x = (uu - K[0, 2]) * zz / K[0, 0]
    y = (vv - K[1, 2]) * zz / K[1, 1]
    ones = np.ones_like(x)
    pts_cam = np.stack([x, y, zz, ones], axis=1)
    pts_world = (pose @ pts_cam.T).T[:, :3]
    colors = color_bgr[vv, uu][:, ::-1].astype(np.float64) / 255.0
    return pts_world, colors


def fuse_scene_points(
    scene_path: Path,
    frame_stride: int = 20,
    pixel_stride: int = 4,
    max_frames: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    color_dir = scene_path / "color"
    depth_dir = scene_path / "depth"
    pose_dir = scene_path / "pose"
    if not color_dir.is_dir():
        raise FileNotFoundError(f"Missing color/ under {scene_path}")
    color_files = sorted(color_dir.glob("*.jpg"))
    if not color_files:
        raise FileNotFoundError(f"No color frames in {color_dir}")
    K = load_intrinsic(scene_path)
    selected = color_files[:: max(1, frame_stride)][:max_frames]
    all_pts, all_cols = [], []
    for cf in selected:
        stem = cf.stem
        df = depth_dir / f"{stem}.png"
        pf = pose_dir / f"{stem}.txt"
        if not df.exists() or not pf.exists():
            continue
        color = cv2.imread(str(cf))
        depth = cv2.imread(str(df), cv2.IMREAD_UNCHANGED)
        if color is None or depth is None:
            continue
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        pts, cols = backproject_frame(color, depth, K, load_pose(pf), stride=pixel_stride)
        if len(pts):
            all_pts.append(pts)
            all_cols.append(cols)
    if not all_pts:
        raise RuntimeError(f"No valid depth frames for {scene_path.name}")
    return np.vstack(all_pts), np.vstack(all_cols)


def render_topdown_bev(
    points: np.ndarray,
    colors: np.ndarray,
    out_w: int = 782,
    out_h: int = 881,
    margin: float = 0.05,
) -> np.ndarray:
    """Orthographic projection onto X–Z (Y-up, ScanNet convention)."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ix, iz = 0, 2
    horiz, vert, height = x, z, y
    xl, xh = np.percentile(horiz, 1), np.percentile(horiz, 99)
    zl, zh = np.percentile(vert, 1), np.percentile(vert, 99)
    dx, dz = xh - xl, zh - zl
    xl -= dx * margin
    xh += dx * margin
    zl -= dz * margin
    zh += dz * margin
    u = ((horiz - xl) / max(xh - xl, 1e-6) * (out_w - 1)).astype(np.int32)
    v = ((vert - zl) / max(zh - zl, 1e-6) * (out_h - 1)).astype(np.int32)
    u = np.clip(u, 0, out_w - 1)
    v = np.clip(v, 0, out_h - 1)
    order = np.argsort(height)
    canvas = np.ones((out_h, out_w, 3), dtype=np.uint8) * 255
    for idx in order:
        canvas[out_h - 1 - v[idx], u[idx]] = (colors[idx] * 255).astype(np.uint8)
    return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)


def scene_ids_from_dataset(path: Path) -> list[str]:
    with open(path) as f:
        data = json.load(f)
    return sorted({q["scene_id"] for q in data if "scene_id" in q})


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BEV maps from ScanNet RGB-D.")
    parser.add_argument("--scannet_root", required=True, help="ScanNet root (scene0000_00/...)")
    parser.add_argument("--output_dir", default="bev_maps", help="Output directory")
    parser.add_argument("--scenes", nargs="*", help="Scene IDs to process")
    parser.add_argument("--from_dataset", help="JSON dataset; extract unique scene_id values")
    parser.add_argument("--frame_stride", type=int, default=20, help="Use every N-th color frame")
    parser.add_argument("--pixel_stride", type=int, default=4, help="Depth subsample stride")
    parser.add_argument("--max_frames", type=int, default=80, help="Max frames fused per scene")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    scannet_root = Path(args.scannet_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_dataset:
        scenes = scene_ids_from_dataset(Path(args.from_dataset))
    elif args.scenes:
        scenes = args.scenes
    else:
        parser.error("Provide --scenes and/or --from_dataset")

    ok, fail = 0, 0
    for scene_id in tqdm(scenes, desc="BEV"):
        out_path = out_dir / f"{scene_id}.jpg"
        if args.skip_existing and out_path.exists():
            ok += 1
            continue
        scene_path = scannet_root / scene_id
        try:
            pts, cols = fuse_scene_points(
                scene_path,
                frame_stride=args.frame_stride,
                pixel_stride=args.pixel_stride,
                max_frames=args.max_frames,
            )
            bev = render_topdown_bev(pts, cols)
            cv2.imwrite(str(out_path), bev)
            ok += 1
        except Exception as exc:
            fail += 1
            print(f"[fail] {scene_id}: {exc}", file=sys.stderr)

    print(f"Done: {ok} saved, {fail} failed -> {out_dir.resolve()}")
    if ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
