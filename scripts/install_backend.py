"""Build only the bundled radiative extension, without downloading model weights."""
from pathlib import Path
import subprocess
import sys
import torch
from torch.utils.cpp_extension import CUDA_HOME


if __name__ == "__main__":
    if CUDA_HOME is None or torch.version.cuda is None:
        raise SystemExit("Install a CUDA-enabled PyTorch build and a compatible CUDA toolkit with nvcc first.")
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "--force-reinstall", str(root/"third_party/ray_gaussian")], check=True)
