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

Generate `I_bev` with [BundleFusion](https://github.com/niessner/BundleFusion) (Appendix B.1): reconstruct a scene point cloud from ScanNet RGB-D, then render a fixed top-down orthographic view. Save as `bev_maps/{scene_id}.jpg`.

Details: [docs/BEV.md](docs/BEV.md). Override path with `export BEV_MAPS_DIR=./bev_maps`.

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
