# Optional container image. On a rented box, scripts/setup_server.sh is simpler
# and faster; this exists for reproducibility and for schedulers that want an
# image.
#
#   docker build -t mpsae --build-arg CUDA_TAG=12.8.0 .
#   docker run --gpus all --env-file .env -v "$PWD/work:/work" \
#       -e MPSAE_WORKSPACE=/work mpsae
#
# CUDA_TAG and TORCH_INDEX must agree, and the host driver must be new enough:
# CUDA 13.x needs driver >= 580, 12.8 needs >= 570.
#   default (13.0 / cu132): Ampere through Blackwell, current torch
#   --build-arg CUDA_TAG=12.8.0 --build-arg TORCH_INDEX=.../cu128  driver 570-579
#   --build-arg CUDA_TAG=12.6.3 --build-arg TORCH_INDEX=.../cu126  V100 and older,
#       which CUDA 13 dropped (it compiles only for sm_75 and above)
ARG CUDA_TAG=13.0.0
FROM nvidia/cuda:${CUDA_TAG}-devel-ubuntu22.04

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu132
# A100, L40S/4090, H100, Blackwell datacentre, Blackwell consumer. Every entry
# must be an architecture the chosen CUDA_TAG still supports.
ARG TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0;10.0;12.0"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
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
