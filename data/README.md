# Ambi3D Data Files

Annotation files are hosted on Hugging Face, not in this repository.

**Dataset:** [jiayuttkx/ambi3d](https://huggingface.co/datasets/jiayuttkx/ambi3d)

```bash
bash scripts/download_data.sh
```

| HF file | Symlink (for eval scripts) | Description |
|---------|---------------------------|-------------|
| `ambi3d_test.json` | `ambitest.json` | Test split (2,131) |
| `ambi3d_all.json` | `ambi3d_v8.json` | Full benchmark (22,081) |
| `ambi3d_train.json` | `ambitrain.json` | Train split |

Manual download:

```bash
pip install huggingface_hub
huggingface-cli download jiayuttkx/ambi3d \
  ambi3d_train.json ambi3d_test.json ambi3d_all.json \
  --local-dir data/
cd data && ln -sf ambi3d_test.json ambitest.json \
  && ln -sf ambi3d_train.json ambitrain.json \
  && ln -sf ambi3d_all.json ambi3d_v8.json
```
