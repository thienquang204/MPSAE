# Optional container image. On a rented box, scripts/setup_server.sh is simpler
# and faster; this exists for reproducibility and for schedulers that want an
# image.
#
#   docker build -t mpsae --build-arg CUDA_TAG=12.8.0 .
#   docker run --gpus all --env-file .env -v "$PWD/work:/work" \
#       -e MPSAE_WORKSPACE=/work mpsae
#
# Pick CUDA_TAG for the GPU: 12.8.0 for Blackwell (RTX 5090, RTX PRO 6000),
# 12.4.1 is enough for H100 and Ada.
ARG CUDA_TAG=12.8.0
FROM nvidia/cuda:${CUDA_TAG}-devel-ubuntu22.04

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
# Kernels are compiled for H100, Blackwell datacentre and Blackwell consumer.
ENV TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0" \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    MPSAE_WORKSPACE=/work

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        build-essential pkg-config git curl ca-certificates \
        libturbojpeg0-dev libopencv-dev \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /opt/mpsae
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip wheel setuptools \
    && python3 -m pip install --index-url "${TORCH_INDEX}" torch torchvision \
    && python3 -m pip install -r requirements.txt

# FFCV has no wheels and is only needed by the CSRv2 embedding recipe, so a
# build failure must not fail the image: the ff2048 backbone still works.
RUN python3 -m pip install ffcv || \
    echo "ffcv unavailable; run with --backbones ff2048" > /opt/mpsae/FFCV_MISSING

COPY mpsae/ ./mpsae/
COPY csrc/ ./csrc/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

VOLUME ["/work"]
ENTRYPOINT ["python3", "-m", "mpsae"]
CMD ["run"]
