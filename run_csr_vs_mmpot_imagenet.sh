#!/usr/bin/env bash
set -Eeuo pipefail

# Launcher for FF2048, the Appendix E.1 ResNet-18 study, and legacy ablations.
# The legacy filename is retained for command compatibility.
#
# Usage:
#   ./run_csr_vs_mmpot_imagenet.sh /path/to/CSR-precompute-embeds
#   BACKBONES=resnet50 DATA_ROOT=/path/to/CSR-precompute-embeds ./run_csr_vs_mmpot_imagenet.sh
#   # For the legacy protocol, DATA_ROOT may instead point to ImageNet.
#
# Protocol validation in the Python entry point prevents mixing ResNet-50
# FF2048, timm ResNet-18 Appendix E.1, and torchvision ablation sources.
# Set BACKBONES to a comma-separated subset, or pass one --backbone argument.
# Other arguments after DATA_ROOT are forwarded unchanged to every Python run.
# Missing dependencies are installed only when INSTALL_DEPS=1; the Docker image
# is already prepared.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# A positional data root always wins. The all-experiments container passes
# it positionally while the same name may also be present in the sourced .env.
if [[ $# -gt 0 && "$1" != -* ]]; then
    DATA_ROOT="$1"
    shift
fi

if [[ -z "${DATA_ROOT:-}" ]]; then
    echo "Usage: $0 /path/to/imagenet [experiment arguments...]" >&2
    echo "   or: DATA_ROOT=/path/to/imagenet $0 [experiment arguments...]" >&2
    exit 2
fi

DATA_BACKEND="${DATA_BACKEND:-imagefolder}"
BACKBONES="${BACKBONES:-resnet50}"
CACHE_DIR="${CACHE_DIR:-$SCRIPT_DIR/runs/three_method_ablation/cache}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/runs/three_method_ablation}"
WEIGHTS_CACHE="${WEIGHTS_CACHE:-$SCRIPT_DIR/weights}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
FAISS_GPU="${FAISS_GPU:-1}"
FAISS_GPU_DEVICE="${FAISS_GPU_DEVICE:-}"
FAISS_TEMP_MEMORY_MIB="${FAISS_TEMP_MEMORY_MIB:-512}"
KNN_QUERY_BATCH="${KNN_QUERY_BATCH:-32}"
SPARSE_KNN_QUERY_BATCH="${SPARSE_KNN_QUERY_BATCH:-$KNN_QUERY_BATCH}"
SPARSE_GALLERY_CHUNK="${SPARSE_GALLERY_CHUNK:-65536}"
RETRIEVAL_WARMUP="${RETRIEVAL_WARMUP:-3}"
RETRIEVAL_REPEATS="${RETRIEVAL_REPEATS:-5}"
SPARSE_SCORE_BUFFER_MIB="${SPARSE_SCORE_BUFFER_MIB:-256}"
SPARSE_GPU_MEMORY_FRACTION="${SPARSE_GPU_MEMORY_FRACTION:-0.8}"
AGGREGATE_RESULTS="${AGGREGATE_RESULTS:-1}"

# Treat a forwarded --backbone as the single-backbone shorthand while keeping
# every other argument intact for each run in the sweep.
forwarded_args=()
cli_backbone=""
while (( $# > 0 )); do
    case "$1" in
        --backbone)
            if (( $# < 2 )); then
                echo "Error: --backbone requires a value." >&2
                exit 2
            fi
            cli_backbone="$2"
            shift 2
            ;;
        --backbone=*)
            cli_backbone="${1#*=}"
            shift
            ;;
        --cache-dir)
            if (( $# < 2 )); then
                echo "Error: --cache-dir requires a value." >&2
                exit 2
            fi
            CACHE_DIR="$2"
            shift 2
            ;;
        --cache-dir=*)
            CACHE_DIR="${1#*=}"
            shift
            ;;
        --output-dir)
            if (( $# < 2 )); then
                echo "Error: --output-dir requires a value." >&2
                exit 2
            fi
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --output-dir=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        *)
            forwarded_args+=("$1")
            shift
            ;;
    esac
done
if [[ -n "$cli_backbone" ]]; then
    BACKBONES="$cli_backbone"
fi

IFS=',' read -r -a requested_backbones <<< "$BACKBONES"
backbone_list=()
for requested in "${requested_backbones[@]}"; do
    backbone="${requested//[[:space:]]/}"
    case "$backbone" in
        resnet18|resnet50) backbone_list+=("$backbone") ;;
        "") ;;
        *)
            echo "Error: unsupported backbone '$backbone' (use resnet18 or resnet50)." >&2
            exit 2
            ;;
    esac
done
if (( ${#backbone_list[@]} == 0 )); then
    echo "Error: BACKBONES did not contain a supported backbone." >&2
    exit 2
fi

case "$DATA_BACKEND" in
    imagefolder|hf) ;;
    *)
        echo "Error: DATA_BACKEND must be 'imagefolder' or 'hf', not '$DATA_BACKEND'." >&2
        exit 2
        ;;
esac

required_modules=(torch torchvision timm numpy scipy faiss PIL matplotlib ninja)
if [[ "$DATA_BACKEND" == "hf" ]]; then
    required_modules+=(datasets huggingface_hub)
fi

missing_modules=()
for module in "${required_modules[@]}"; do
    if ! "$PYTHON_BIN" -c "import ${module}" >/dev/null 2>&1; then
        missing_modules+=("$module")
    fi
done

case "$FAISS_GPU" in
    1)
        if ! "$PYTHON_BIN" -c "import faiss; raise SystemExit(0 if hasattr(faiss, 'StandardGpuResources') else 1)" >/dev/null 2>&1; then
            missing_modules+=(faiss)
        fi
        ;;
    0)
        echo "Error: GPU retrieval comparisons require FAISS_GPU=1; CPU fallback is disabled." >&2
        exit 2
        ;;
    *)
        echo "Error: FAISS_GPU must be 0 or 1" >&2
        exit 2
        ;;
esac

if (( ${#missing_modules[@]} > 0 )); then
    if [[ "$FAISS_GPU" == "1" ]] && printf '%s\n' "${missing_modules[@]}" | grep -qx faiss; then
        echo "Error: CUDA FAISS is missing from this image." >&2
        echo "Rebuild it with run_all_experiments_docker.sh; GX10/ARM64 FAISS is compiled at image build time." >&2
        exit 1
    fi
    if [[ "$INSTALL_DEPS" != "1" ]]; then
        echo "Error: missing Python modules: ${missing_modules[*]}" >&2
        echo "Install requirements.txt or rerun with INSTALL_DEPS=1." >&2
        exit 1
    fi
    echo "Installing missing Python dependencies: ${missing_modules[*]}"
    "$PYTHON_BIN" -m pip install --requirement "$SCRIPT_DIR/requirements.txt"

    for module in "${required_modules[@]}"; do
        "$PYTHON_BIN" -c "import ${module}" || {
            echo "Error: '$module' is unavailable after dependency installation." >&2
            exit 1
        }
    done
fi

if [[ "$FAISS_GPU" == "1" ]]; then
    cuda_nvcc="${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}/bin/nvcc"
    if [[ ! -x "$cuda_nvcc" ]] && ! command -v nvcc >/dev/null 2>&1; then
        echo "Error: sparse GPU retrieval requires nvcc and a CUDA toolkit compatible with PyTorch." >&2
        echo "Use the project's CUDA devel image or configure CUDA_HOME before starting." >&2
        exit 1
    fi
    "$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot access CUDA'" || {
        echo "Error: FAISS_GPU=1 requires a CUDA-capable PyTorch runtime and visible NVIDIA GPU." >&2
        exit 1
    }
    "$PYTHON_BIN" -c "import faiss; assert hasattr(faiss, 'StandardGpuResources'), 'FAISS is CPU-only'" || {
        echo "Error: CUDA-enabled FAISS is required; rebuild the project image." >&2
        exit 1
    }
fi

mkdir -p "$CACHE_DIR" "$OUTPUT_DIR/logs" "$WEIGHTS_CACHE"

case "$AGGREGATE_RESULTS" in
    1)
        [[ -f "$SCRIPT_DIR/aggregate_backbone_results.py" ]] || {
            echo "Error: aggregate_backbone_results.py is missing." >&2
            exit 1
        }
        # An old marker must not make an interrupted rerun look complete.
        rm -f "$OUTPUT_DIR/RUN_COMPLETE.json"
        ;;
    0) ;;
    *)
        echo "Error: AGGREGATE_RESULTS must be 0 or 1." >&2
        exit 2
        ;;
esac

echo "Running adaptive-representation experiment"
echo "  backend: $DATA_BACKEND"
echo "  data:    $DATA_ROOT"
echo "  models:  ${backbone_list[*]}"
echo "  cache:   $CACHE_DIR/<backbone>"
echo "  output:  $OUTPUT_DIR/<backbone>"
echo "  retrieval: CUDA FAISS + CUDA sparse inverted index (no CPU fallback)"

for backbone in "${backbone_list[@]}"; do
    command=(
        "$PYTHON_BIN" "$SCRIPT_DIR/csr_vs_mmpot_imagenet.py"
        --data-root "$DATA_ROOT"
        --data-backend "$DATA_BACKEND"
        --backbone "$backbone"
        --cache-dir "$CACHE_DIR/$backbone"
        --output-dir "$OUTPUT_DIR/$backbone"
        --weights-cache "$WEIGHTS_CACHE"
        --knn-query-batch "$KNN_QUERY_BATCH"
        --sparse-knn-query-batch "$SPARSE_KNN_QUERY_BATCH"
        --sparse-gallery-chunk "$SPARSE_GALLERY_CHUNK"
        --retrieval-warmup "$RETRIEVAL_WARMUP"
        --retrieval-repeats "$RETRIEVAL_REPEATS"
        --sparse-score-buffer-mib "$SPARSE_SCORE_BUFFER_MIB"
        --sparse-gpu-memory-fraction "$SPARSE_GPU_MEMORY_FRACTION"
        --faiss-temp-memory-mib "$FAISS_TEMP_MEMORY_MIB"
    )
    command+=(--faiss-gpu)
    if [[ -n "$FAISS_GPU_DEVICE" ]]; then
        command+=(--faiss-gpu-device "$FAISS_GPU_DEVICE")
    fi
    command+=("${forwarded_args[@]}")

    echo
    echo "[$backbone] starting experiment"
    printf '  command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    "${command[@]}" 2>&1 | tee -a "$OUTPUT_DIR/logs/${backbone}_representation_comparison.log"
done

if [[ "$AGGREGATE_RESULTS" == "1" ]]; then
    backbone_csv="$(IFS=,; echo "${backbone_list[*]}")"
    echo
    echo "Aggregating tables, plots, manifest, and portable results bundle..."
    "$PYTHON_BIN" "$SCRIPT_DIR/aggregate_backbone_results.py" \
        "$OUTPUT_DIR" \
        --backbones "$backbone_csv"
    echo "Collect the ZIP named in: $OUTPUT_DIR/RUN_COMPLETE.json"
fi

echo "Completed representation comparison: ${backbone_list[*]}"
