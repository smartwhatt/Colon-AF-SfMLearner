# =============================================================================
# AF-SfMLearner C3VDv2 Training Image
# Base: PyTorch 2.8.0 with CUDA 12.9 and cuDNN 9
# 
# Build: 
# docker build -t smartwatt/af-sfmlearner . 
# 
# Run (interactive): 
# docker run --gpus device=1 --rm -it \ 
# -v /path/to/C3VDv2:/data/C3VDv2 \ 
# -v /path/to/repo:/workspace \ 
# -v /path/to/logs:/workspace/logs \ 
# smartwatt/af-sfmlearner bash # 
# Run (direct fine-tuning): 
# docker run --gpus device=1 --rm \ 
# -v /path/to/C3VDv2:/data/C3VDv2 \ 
# -v /path/to/repo:/workspace \ 
# -v /path/to/logs:/workspace/logs \ 
# smartwatt/af-sfmlearner \ # bash scripts/run_finetune_c3vdv2.sh 
#
# =============================================================================
FROM --platform=linux/amd64 pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    curl \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.11 default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# ---------------------------------------------------------------------------
# Install uv (fast python package manager)
# ---------------------------------------------------------------------------
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir uv

# ---------------------------------------------------------------------------
# Set working directory
# ---------------------------------------------------------------------------
WORKDIR /workspace

# ---------------------------------------------------------------------------
# Copy only dependency files first (for Docker caching)
# ---------------------------------------------------------------------------
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv
RUN uv sync --no-dev --python 3.11

# ---------------------------------------------------------------------------
# Install CUDA-compatible PyTorch separately
# ---------------------------------------------------------------------------
# RUN uv pip install \
#     torch==2.1.2+cu118 \
#     torchvision==0.16.2+cu118 \
#     --index-url https://download.pytorch.org/whl/cu118

# ---------------------------------------------------------------------------
# Runtime environment settings
# ---------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4

CMD ["bash"]