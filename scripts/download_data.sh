#!/usr/bin/env bash
# Download Ambi3D annotations from Hugging Face into data/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${1:-$ROOT/data}"
mkdir -p "$DATA_DIR"

HF_REPO="jiayuttkx/ambi3d"

if command -v hf >/dev/null 2>&1; then
  hf download "$HF_REPO" ambi3d_train.json ambi3d_test.json ambi3d_all.json \
    --repo-type dataset --local-dir "$DATA_DIR"
else
  pip install -q huggingface_hub
  DATA_DIR="$DATA_DIR" python <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download

dest = os.environ["DATA_DIR"]
repo = "jiayuttkx/ambi3d"
for name in ("ambi3d_train.json", "ambi3d_test.json", "ambi3d_all.json"):
    path = hf_hub_download(repo_id=repo, filename=name, repo_type="dataset")
    shutil.copy(path, os.path.join(dest, name))
    print(f"Downloaded {name}")
PY
fi

cd "$DATA_DIR"
ln -sf ambi3d_test.json ambitest.json
ln -sf ambi3d_train.json ambitrain.json
ln -sf ambi3d_all.json ambi3d_v8.json

echo "Downloaded from https://huggingface.co/datasets/$HF_REPO"
ls -lh "$DATA_DIR"/*.json
