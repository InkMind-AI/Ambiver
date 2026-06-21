# AmbiVer: 3D Instruction Ambiguity Detection

Official code for [*3D Instruction Ambiguity Detection*](https://arxiv.org/abs/2601.05991) (arXiv:2601.05991).


---

## Install

```bash
git clone https://github.com/InkMind-AI/Ambiver.git && cd Ambiver

conda create -n ambiver python=3.10 -y && conda activate ambiver
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install -e .
```

You also need [ScanNet](http://www.scan-net.org/) and [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO). We recommend **two conda envs** (Qwen3-VL and GroundingDINO often conflict). See [docs/SETUP.md](docs/SETUP.md).

---

## Download data

Annotations are hosted on Hugging Face:

```bash
bash scripts/download_data.sh
```

Dataset: [jiayuttkx/ambi3d](https://huggingface.co/datasets/jiayuttkx/ambi3d)

**BEV maps are not included.** Generate top-down bird's-eye views from ScanNet before evaluation ([docs/BEV.md](docs/BEV.md)).

```bash
python scripts/generate_bev_maps.py \
  --scannet_root "$SCANNET_ROOT" \
  --output_dir ./bev_maps \
  --from_dataset data/ambitest.json
```

---

## Run

```bash
export SCANNET_ROOT=/path/to/scannet
export GROUNDING_DINO_ROOT=/path/to/GroundingDINO
export GROUNDINGDINO_ENV=/path/to/conda/envs/groundingdino
conda activate ambiver

python scripts/run_ambi3d_eval.py \
  --dataset data/ambitest.json \
  --scannet_root "$SCANNET_ROOT" \
  --output_dir /path/to/output/ambitest \
  --no_vis --target_frames 20
```


---



## Citation

```bibtex
@misc{ding20263dinstructionambiguitydetection
      title={3D Instruction Ambiguity Detection}, 
      author={Jiayu Ding and Haoran Tang and Hongbo Jin and Wei Gao and Ge Li},
      year={2026},
      eprint={2601.05991},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.05991}, 
} 
```


