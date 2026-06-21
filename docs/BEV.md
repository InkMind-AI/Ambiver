# Bird's-Eye View (BEV) Maps

AmbiVer requires a **precomputed top-down BEV image** per ScanNet scene (`I_bev` in the paper). BEV maps are **not** shipped with this repository; you must generate them locally from ScanNet.

Paper: [3D Instruction Ambiguity Detection](https://arxiv.org/abs/2601.05991) — §3.2 *Global Feature Acquisition*, Appendix B.1.

---

## Principle (from the paper)

AmbiVer's perception stage builds **global spatial context** before instance-level reasoning:

1. **Reconstruct a scene point cloud** `P` from the egocentric RGB-D stream `V` and camera poses `E` using a reconstruction pipeline `R` (Appendix B.1 uses [BundleFusion](https://github.com/niessner/BundleFusion)).
2. **Orthographic top-down projection** — project `P` into a 2D BEV image `I_bev` with a fixed overhead extrinsic `E_top`:

   `I_bev = T(P, E_top)`

The BEV gives the VLM a compact, allocentric layout of the scene. It is bundled with local instance crops in the **Dossier** for zero-shot ambiguity adjudication (§3.3).

---

## Expected file layout

Place generated maps under `./bev_maps/` (or set `BEV_MAPS_DIR`):

| File | Role |
|------|------|
| `{scene_id}.jpg` | Color BEV (primary input to the evaluator) |
| `{scene_id}_color.jpg` | Optional alias for color BEV |
| `{scene_id}.png` | Grayscale BEV (optional) |

Example: `bev_maps/scene0000_00.jpg`

Evaluation **skips** scenes without a valid BEV (no placeholder is used).

---

## Option A — Lightweight script (included)

We provide a ScanNet RGB-D fusion script that follows the same idea as the paper (multi-view depth fusion → top-down projection) without BundleFusion:

```bash
export SCANNET_ROOT=/path/to/scannet   # scene0000_00/color, depth, pose, intrinsic/

# All scenes referenced in Ambi3D (after download_data.sh):
python scripts/generate_bev_maps.py \
  --scannet_root "$SCANNET_ROOT" \
  --output_dir ./bev_maps \
  --from_dataset data/ambitest.json

# Or specific scenes:
python scripts/generate_bev_maps.py \
  --scannet_root "$SCANNET_ROOT" \
  --output_dir ./bev_maps \
  --scenes scene0000_00 scene0001_00
```

**Pipeline:**
- Read RGB, depth, and 4×4 pose per frame (ScanNet layout).
- Back-project depth to 3D world points; subsample frames for speed.
- Project onto the ground plane (X–Z, Y-up), paint by height order, save JPEG.

This is sufficient for reproduction. For closest match to the paper's offline pipeline, use Option B.

---

## Option B — Paper pipeline (BundleFusion)

Appendix B.1:

1. Run **BundleFusion** on each scene's color/depth/pose sequence → unified point cloud `P`.
2. Render `I_bev` with a **fixed top-down orthographic camera** `E_top` (same viewpoint for all scenes).

Refer to the [BundleFusion repository](https://github.com/niessner/BundleFusion) for installation and ScanNet preprocessing. Export the rendered top-down RGB image as `{scene_id}.jpg` under `bev_maps/`.

---

## Option C — ScanNet mesh (if you have `_vh_clean_2.ply`)

If you downloaded ScanNet **mesh** scans (`scans/{scene_id}/{scene_id}_vh_clean_2.ply`):

1. Apply the scene `axisAlignment` matrix from `{scene_id}.txt`.
2. Orthographically project mesh vertices onto the floor plane (top-down scatter or render).

Any method that yields a consistent overhead layout image is acceptable as long as filenames match the table above.

---

## Verify

```bash
ls bev_maps/scene0000_00.jpg
export BEV_MAPS_DIR=./bev_maps
python scripts/run_ambi3d_eval.py --dataset data/ambitest.json \
  --scannet_root "$SCANNET_ROOT" --output_dir /tmp/ambitest --limit 1 --no_vis
```

If BEV is missing, the eval script prints `Skipping scene …: no BEV`.
