# Reproducing AmbiVer Results

Setup: [SETUP.md](SETUP.md). Pipeline: zero-shot **Qwen3-VL-8B-Instruct**, no LoRA.

## Reference metrics

| Split | JSON | N | Accuracy | F1 |
|-------|------|---|----------|-----|
| Test | `data/ambitest.json` | 2,131 | 0.610 | 0.611 |
| Full | `data/ambi3d_v8.json` | 22,081 | 0.621 | 0.616 |

---

## Test set (2,131 questions)

```bash
export SCANNET_ROOT=/path/to/scannet
export GROUNDING_DINO_ROOT=/path/to/GroundingDINO
export GROUNDINGDINO_ENV=/path/to/conda/envs/groundingdino

CUDA_VISIBLE_DEVICES=0 python scripts/run_ambi3d_eval.py \
  --dataset data/ambitest.json \
  --scannet_root "$SCANNET_ROOT" \
  --output_dir /path/to/output/ambitest \
  --no_vis --target_frames 20 --checkpoint_every 500
```

---

## Full benchmark (22,081 questions)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_ambi3d_eval.py \
  --dataset data/ambi3d_v8.json \
  --scannet_root "$SCANNET_ROOT" \
  --output_dir /path/to/output/full \
  --no_vis --target_frames 20 --checkpoint_every 1000 --resume
```

### Dual-GPU parallel

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_ambi3d_eval.py \
  --dataset data/ambi3d_v8.json --scannet_root "$SCANNET_ROOT" \
  --output_dir /path/to/output/full --no_vis --target_frames 20 \
  --worker_id 0 --num_workers 2 --checkpoint_every 1000 &

CUDA_VISIBLE_DEVICES=1 python scripts/run_ambi3d_eval.py \
  --dataset data/ambi3d_v8.json --scannet_root "$SCANNET_ROOT" \
  --output_dir /path/to/output/full --no_vis --target_frames 20 \
  --worker_id 1 --num_workers 2 --checkpoint_every 1000 &
wait
```

Outputs per worker: `/path/to/output/full/worker_{0,1}/`.

---

## Output files

| File | Content |
|------|---------|
| `metrics.json` | Accuracy, F1, per-type stats |
| `predictions.json` | Per-question predictions |
| `detailed_results.json` | Full results (for `--resume`) |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `BEV image not found` | Set `BEV_MAPS_DIR` to `./bev_maps` |
| Detection subprocess fails | Set `GROUNDINGDINO_ENV` |
| CUDA OOM | Reduce `--target_frames` to 16 |
