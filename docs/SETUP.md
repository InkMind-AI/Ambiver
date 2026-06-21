# Environment Setup

## 1. Clone & download data

```bash
git clone https://github.com/InkMind-AI/Ambiver.git
cd Ambiver
bash scripts/download_data.sh
```

Dataset: [jiayuttkx/ambi3d](https://huggingface.co/datasets/jiayuttkx/ambi3d)

## 2. Python environment

```bash
conda create -n ambiver python=3.10 -y && conda activate ambiver
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install -e .
```

## 3. ScanNet

Request access at [ScanNet](http://www.scan-net.org/) and download scenes:

```bash
export SCANNET_ROOT=/path/to/scannet   # contains scene0000_00/, scene0001_00/, ...
```

## 4. GroundingDINO

Use a **separate conda env** for detection (recommended):

```bash
conda create -n groundingdino python=3.10 -y && conda activate groundingdino
bash scripts/install_groundingdino.sh ./third_party
export GROUNDING_DINO_ROOT=$(pwd)/third_party/GroundingDINO
export GROUNDINGDINO_ENV=/path/to/conda/envs/groundingdino
```

Evaluation calls GroundingDINO via subprocess; the `ambiver` env does not need it installed.

Single-env alternative: add `--use_batch_detector` to the eval script.

## 5. BEV maps (required, user-generated)

AmbiVer needs a precomputed **bird's-eye view (BEV)** image per scene (`I_bev`, paper §3.2). They are **not** shipped in this repo.

**Principle:** fuse multi-view RGB-D into a 3D point cloud, then orthographically project with a fixed top-down camera (Appendix B.1 uses [BundleFusion](https://github.com/niessner/BundleFusion)).

**Quick start** (included script):

```bash
export SCANNET_ROOT=/path/to/scannet
python scripts/generate_bev_maps.py \
  --scannet_root "$SCANNET_ROOT" \
  --output_dir ./bev_maps \
  --from_dataset data/ambitest.json
```

Full details and alternatives: [docs/BEV.md](docs/BEV.md). Set `export BEV_MAPS_DIR=./bev_maps` if using a custom path.

## 6. Qwen3-VL-8B

Auto-downloads from HuggingFace on first run (`Qwen/Qwen3-VL-8B-Instruct`).

## Environment variables

| Variable | Description |
|----------|-------------|
| `SCANNET_ROOT` | ScanNet root (required) |
| `GROUNDING_DINO_ROOT` | GroundingDINO clone |
| `GROUNDINGDINO_ENV` | Conda env for detection subprocess |
| `BEV_MAPS_DIR` | Precomputed BEV images (default: `./bev_maps`) |
| `CUDA_VISIBLE_DEVICES` | GPU selection |
