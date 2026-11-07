# Optional container image. On a rented box, scripts/setup_server.sh is simpler
# and faster; this exists for reproducibility and for schedulers that want an
# image.
#
#   docker build -t mpsae .
#   docker compose run --rm mpsae
#   docker compose run --rm mpsae --config configs/thesis.yaml plan
# The default command runs the configured ImageNet + frozen-transfer campaign.
#
# CUDA_TAG and TORCH_INDEX must agree, and the host driver must be new enough:
# CUDA 13.x needs driver >= 580, 12.8 needs >= 570.
#   default (13.0 / cu130): Ampere through Blackwell
#   --build-arg CUDA_TAG=12.8.0 --build-arg TORCH_INDEX=.../cu128  driver 570-579
#   --build-arg CUDA_TAG=12.6.3 --build-arg TORCH_INDEX=.../cu126  V100 and older,
#       which CUDA 13 dropped (it compiles only for sm_75 and above)
ARG CUDA_TAG=13.0.0
FROM nvidia/cuda:${CUDA_TAG}-devel-ubuntu24.04

ARG CUDA_TAG
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
# A100, L40S/4090, H100, Blackwell datacentre, Blackwell consumer. Every entry
# must be an architecture the chosen CUDA_TAG still supports.
ARG TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0;10.0;12.0"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    PATH="/opt/venv/bin:${PATH}" \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    MPSAE_WORKSPACE=/work \
    MPSAE_CONFIG=configs/thesis.yaml

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv \
        build-essential pkg-config git curl ca-certificates \
        libturbojpeg0-dev libopencv-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

WORKDIR /opt/mpsae
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip wheel setuptools \
    && python3 -m pip install --index-url "${TORCH_INDEX}" torch torchvision \
    && python3 -m pip install -r requirements.txt ninja

# The default campaign includes rn18: FFCV and its GPU normalization dependency
# are required. A failed dependency build must fail the image immediately.
RUN case "${CUDA_TAG%%.*}" in \
        12) CUPY_PACKAGE=cupy-cuda12x ;; \
        13) CUPY_PACKAGE=cupy-cuda13x ;; \
        *) echo "Unsupported CUDA major in CUDA_TAG=${CUDA_TAG}" >&2; exit 1 ;; \
    esac \
    && python3 -m pip install "${CUPY_PACKAGE}" ffcv \
    && python3 -c "import cupy, ffcv"

COPY mpsae/ ./mpsae/
COPY csrc/ ./csrc/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY configs/ ./configs/

VOLUME ["/work"]
ENTRYPOINT ["python3", "-m", "mpsae"]
CMD ["--config", "configs/thesis.yaml", "run"]
