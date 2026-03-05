# =============================================================================
# AF-SfMLearner C3VDv2 Training Image
# =============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

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
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

RUN python -m pip install --upgrade pip

# ---------------------------------------------------------------------------
# Set working directory
# ---------------------------------------------------------------------------
WORKDIR /workspace

# ---------------------------------------------------------------------------
# Copy only dependency files first (for Docker caching)
# ---------------------------------------------------------------------------
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv
RUN uv sync --no-dev

# ---------------------------------------------------------------------------
# Install CUDA-compatible PyTorch separately
# ---------------------------------------------------------------------------
RUN uv pip install \
    torch==2.1.2+cu118 \
    torchvision==0.16.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# ---------------------------------------------------------------------------
# Runtime environment settings
# ---------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4

CMD ["bash"]