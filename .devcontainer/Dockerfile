# Base image with Python 3.10
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and PyTorch3D
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir pytorch3d

# Install additional Python dependencies
RUN pip install --no-cache-dir \
    einops>=0.3 \
    En-transformer>=0.2.3 \
    invariant-point-attention \
    mdtraj>=1.8 \
    numpy \
    proDy \
    requests \
    sidechainnet \
    transformers \
    tqdm \
    biopython \
    mp-nerf>=0.1.5
