#!/usr/bin/env bash
# Clone GroundingDINO and prepare weight directory.
set -euo pipefail

INSTALL_DIR="${1:-./third_party}"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

if [ ! -d GroundingDINO ]; then
  git clone https://github.com/IDEA-Research/GroundingDINO.git
fi

cd GroundingDINO
pip install -e .
mkdir -p weights

if [ ! -f weights/groundingdino_swint_ogc.pth ]; then
  echo "Download weights manually:"
  echo "  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
  echo "Save to: $(pwd)/weights/groundingdino_swint_ogc.pth"
fi

echo "Done. Export:"
echo "  export GROUNDING_DINO_ROOT=$(pwd)"
