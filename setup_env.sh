#!/usr/bin/env bash
# Creates ./.venv next to this script, installs everything, and registers a Jupyter kernel.
# Usage (macOS / Linux):   bash setup_env.sh
# Optional overrides:      PYTHON=python3.11 CUDA_TAG=cu128 bash setup_env.sh
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
CUDA_TAG="${CUDA_TAG:-cu126}"        # cu126 works with NVIDIA driver >= 525; use cu128 for RTX 50-series
KERNEL_NAME="cmpt819-deepfake"

echo ">> Using $($PY --version)  (3.10 – 3.12 recommended)"
$PY -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

if command -v nvidia-smi >/dev/null 2>&1; then
    echo ">> NVIDIA GPU found -> installing CUDA build ($CUDA_TAG)"
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
elif [[ "$(uname)" == "Darwin" ]]; then
    echo ">> macOS -> default build (uses Apple MPS if available)"
    pip install torch torchvision
else
    echo ">> No GPU -> CPU build"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

pip install -r requirements.txt
pip install --no-deps facenet-pytorch==2.6.0

python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "Python (CMPT819 Deepfake)"

python - <<'EOF'
import torch, torchvision, timm, facenet_pytorch, cv2, sklearn
from facenet_pytorch import MTCNN
MTCNN(device="cpu")   # weights ship inside the package; this confirms they load
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"torch {torch.__version__} | torchvision {torchvision.__version__} | timm {timm.__version__} | device: {dev}")
if dev == "cuda": print("GPU:", torch.cuda.get_device_name(0))
print("Environment OK")
EOF
echo ">> Done. In VS Code / Jupyter pick the kernel: Python (CMPT819 Deepfake)"
