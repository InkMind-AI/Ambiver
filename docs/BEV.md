# Bird's-Eye View (BEV) Maps

AmbiVer requires a precomputed top-down BEV image per ScanNet scene (`I_bev`, paper §3.2). BEV maps are **not** included in this repository.

Paper: [3D Instruction Ambiguity Detection](https://arxiv.org/abs/2601.05991) — §3.2 *Global Feature Acquisition*, Appendix B.1.

---

## Principle

Per Appendix B.1, `I_bev` is generated in two steps:

1. **Point cloud reconstruction** — [BundleFusion](https://github.com/niessner/BundleFusion) aggregates the egocentric RGB-D stream `V` and camera poses `E` into a unified 3D point cloud `P`.
2. **Top-down projection** — orthographically project `P` into `I_bev` with a fixed overhead extrinsic `E_top`:

   `I_bev = T(P, E_top)`

The BEV provides global spatial context for the VLM reasoning stage (§3.3 Dossier).

---

## Generate with BundleFusion

1. Install and build [BundleFusion](https://github.com/niessner/BundleFusion) following its README.
2. For each ScanNet scene, run BundleFusion on the scene's color / depth / pose sequence to obtain point cloud `P`.
3. Render a **fixed top-down orthographic** RGB view of `P` (same `E_top` for all scenes).
4. Save as `{scene_id}.jpg` under `./bev_maps/` (or set `BEV_MAPS_DIR`).

| File | Role |
|------|------|
| `{scene_id}.jpg` | Color BEV (used by evaluation) |
| `{scene_id}_color.jpg` | Optional alias |

Example: `bev_maps/scene0000_00.jpg`

Evaluation skips scenes without a valid BEV.

---

## Verify

```bash
ls bev_maps/scene0000_00.jpg
export BEV_MAPS_DIR=./bev_maps
python scripts/run_ambi3d_eval.py --dataset data/ambitest.json \
  --scannet_root "$SCANNET_ROOT" --output_dir /tmp/ambitest --limit 1 --no_vis
```
