#!/usr/bin/env bash

# Reproduce Morpheus3D on RealFusion15 and Morpheus30:
#
#   coarse -> refine -> texture -> five evaluation metrics
#
# Activate the unified environment and download the model weights first:
#
#   conda activate morpheus3d
#   ./download_weights.sh
#   ./run_reproduction.sh --gpu 0

set -Eeuo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly RUN_TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
readonly PYTHON_BIN="${PYTHON_BIN:-python}"

GPU_ID="0"
OUTPUT_ROOT=""
DRY_RUN=false
DATASETS=()
SCENES=()
CURRENT_STEP="initialization"

usage() {
    cat <<'EOF'
Usage: ./run_reproduction.sh [OPTIONS]

By default, the script reproduces every scene in both realfusion15 and
morpheus30 on physical GPU 0. Each run uses a new timestamped output root.

Options:
  --dataset NAME      Run one dataset. Repeat to select both datasets.
                      Choices: realfusion15, morpheus30.
  --scene NAME        Run one scene. Repeat to select multiple scenes.
                      This option requires exactly one selected dataset.
  --gpu ID            Physical GPU index to expose to the pipeline (default: 0).
  --output-root PATH  Exact output root; it must be absent or empty.
  --dry-run           Print the top-level commands without executing them.
  -h, --help          Show this help message.

Examples:
  ./run_reproduction.sh --gpu 0
  ./run_reproduction.sh --dataset realfusion15 --scene banana --gpu 0
  ./run_reproduction.sh --dataset morpheus30 --scene scene_00 --gpu 1
EOF
}

log() {
    printf '[run_reproduction] %s\n' "$*"
}

die() {
    printf '[run_reproduction] ERROR: %s\n' "$*" >&2
    exit 1
}

on_error() {
    local exit_code=$?
    printf '[run_reproduction] ERROR: failed during %s (exit code %s)\n' \
        "${CURRENT_STEP}" "${exit_code}" >&2
    exit "${exit_code}"
}

trap on_error ERR

require_value() {
    local option="$1"
    local value="${2:-}"
    [[ -n "${value}" ]] || die "${option} requires a value"
}

append_dataset() {
    local dataset="$1"
    case "${dataset}" in
        realfusion15|morpheus30) ;;
        *) die "Unsupported dataset: ${dataset}" ;;
    esac

    if [[ ${#DATASETS[@]} -gt 0 ]]; then
        local existing
        for existing in "${DATASETS[@]}"; do
            [[ "${existing}" == "${dataset}" ]] && return
        done
    fi
    DATASETS+=("${dataset}")
}

resolve_project_path() {
    local path="$1"
    if [[ "${path}" == /* ]]; then
        printf '%s\n' "${path}"
    else
        printf '%s/%s\n' "${PROJECT_ROOT}" "${path}"
    fi
}

print_command() {
    printf '  +'
    printf ' %q' "$@"
    printf '\n'
}

run_command() {
    print_command "$@"
    if [[ "${DRY_RUN}" == false ]]; then
        "$@"
    fi
}

require_file() {
    [[ -f "$1" ]] || die "Required file not found: $1"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            require_value "$1" "${2:-}"
            append_dataset "$2"
            shift 2
            ;;
        --scene)
            require_value "$1" "${2:-}"
            SCENES+=("$2")
            shift 2
            ;;
        --gpu)
            require_value "$1" "${2:-}"
            GPU_ID="$2"
            shift 2
            ;;
        --output-root)
            require_value "$1" "${2:-}"
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
done

if [[ ${#DATASETS[@]} -eq 0 ]]; then
    DATASETS=(realfusion15 morpheus30)
fi

[[ "${GPU_ID}" =~ ^[0-9]+$ ]] || die "--gpu must be one physical GPU index"
if [[ ${#SCENES[@]} -gt 0 && ${#DATASETS[@]} -ne 1 ]]; then
    die "--scene requires exactly one selected --dataset"
fi

if [[ -z "${OUTPUT_ROOT}" ]]; then
    OUTPUT_ROOT="${PROJECT_ROOT}/outputs/reproduction-${RUN_TIMESTAMP}"
else
    OUTPUT_ROOT="$(resolve_project_path "${OUTPUT_ROOT}")"
fi

if [[ -d "${OUTPUT_ROOT}" ]]; then
    first_output_entry="$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)"
    [[ -z "${first_output_entry}" ]] || \
        die "Output root is not empty; use a new path: ${OUTPUT_ROOT}"
elif [[ -e "${OUTPUT_ROOT}" ]]; then
    die "Output root exists and is not a directory: ${OUTPUT_ROOT}"
fi

if [[ "${DRY_RUN}" == false ]]; then
    command -v "${PYTHON_BIN}" >/dev/null 2>&1 || \
        die "Python command not found: ${PYTHON_BIN}"
fi

cd "${PROJECT_ROOT}"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

if [[ "${DRY_RUN}" == false ]]; then
    # Fail before a long training run if setup is incomplete. The detailed
    # download and path setup is handled by download_weights.sh.
    if [[ -n "${TORCH_HOME:-}" ]]; then
        TORCH_CACHE_ROOT="${TORCH_HOME}/hub"
    elif [[ -n "${XDG_CACHE_HOME:-}" ]]; then
        TORCH_CACHE_ROOT="${XDG_CACHE_HOME}/torch/hub"
    else
        [[ -n "${HOME:-}" ]] || die "HOME is unset; cannot locate the PyTorch cache"
        TORCH_CACHE_ROOT="${HOME}/.cache/torch/hub"
    fi

    required_runtime_files=(
        "${PROJECT_ROOT}/diffusion_ckpt/zero123-xl-diffusers-base/model_index.json"
        "${PROJECT_ROOT}/diffusion_ckpt/zero123-xl-diffusers-base/feature_extractor/preprocessor_config.json"
        "${PROJECT_ROOT}/diffusion_ckpt/zero123-xl-diffusers-base/scheduler/scheduler_config.json"
        "${PROJECT_ROOT}/diffusion_ckpt/zero123-xl-diffusers-base/clip_camera_projection/config.json"
        "${PROJECT_ROOT}/diffusion_ckpt/zero123-xl-diffusers-base/clip_camera_projection/diffusion_pytorch_model.fp16.safetensors"
        "${PROJECT_ROOT}/diffusion_ckpt/zero123-xl-diffusers-base/image_encoder/config.json"
        "${PROJECT_ROOT}/diffusion_ckpt/zero123-xl-diffusers-base/image_encoder/model.fp16.safetensors"
        "${PROJECT_ROOT}/diffusion_ckpt/zero123-xl-diffusers-base/unet/config.json"
        "${PROJECT_ROOT}/diffusion_ckpt/zero123-xl-diffusers-base/unet/diffusion_pytorch_model.fp16.safetensors"
        "${PROJECT_ROOT}/diffusion_ckpt/zero123-xl-diffusers-base/vae/config.json"
        "${PROJECT_ROOT}/diffusion_ckpt/zero123-xl-diffusers-base/vae/diffusion_pytorch_model.fp16.safetensors"
        "${PROJECT_ROOT}/diffusion_ckpt/prompt-free-diffusion-base/controlnet/control_sd15_canny_slimmed.safetensors"
        "${PROJECT_ROOT}/diffusion_ckpt/prompt-free-diffusion-base/pfd/diffuser/SD-v1-5.safetensors"
        "${PROJECT_ROOT}/diffusion_ckpt/prompt-free-diffusion-base/pfd/seecoder/seecoder-v1-0.safetensors"
        "${PROJECT_ROOT}/diffusion_ckpt/prompt-free-diffusion/controlnet/control_sd15_canny_slimmed.safetensors"
        "${PROJECT_ROOT}/diffusion_ckpt/prompt-free-diffusion/pfd/diffuser/SD-v1-5.safetensors"
        "${PROJECT_ROOT}/diffusion_ckpt/prompt-free-diffusion/pfd/seecoder/seecoder-v1-0.safetensors"
        "${PROJECT_ROOT}/extern/pfd/pretrained/pfd/vae/sd-v2-0-base-autokl.pth"
        "${PROJECT_ROOT}/clip_ckpt/CLIP-ViT-bigG-14-laion2B-39B-b160k/config.json"
        "${PROJECT_ROOT}/clip_ckpt/CLIP-ViT-bigG-14-laion2B-39B-b160k/pytorch_model.bin.index.json"
        "${PROJECT_ROOT}/clip_ckpt/CLIP-ViT-bigG-14-laion2B-39B-b160k/pytorch_model-00001-of-00002.bin"
        "${PROJECT_ROOT}/clip_ckpt/CLIP-ViT-bigG-14-laion2B-39B-b160k/pytorch_model-00002-of-00002.bin"
        "${PROJECT_ROOT}/metric_ckpt/vit_base_patch8_224.augreg2_in21k_ft_in1k/model.safetensors"
        "${PROJECT_ROOT}/metric_ckpt/ckpt_koniq10k.pt"
        "${PROJECT_ROOT}/metric_ckpt/RN50.pt"
        "${TORCH_CACHE_ROOT}/checkpoints/alexnet-owt-7be5be79.pth"
        "${PROJECT_ROOT}/load/tets/128_tets.npz"
    )
    for required_runtime_file in "${required_runtime_files[@]}"; do
        require_file "${required_runtime_file}"
    done
fi

if [[ "${DRY_RUN}" == false ]]; then
    mkdir -p "${OUTPUT_ROOT}/metrics"
fi

log "Physical GPU: ${GPU_ID} (visible as cuda:0)"
log "Output root: ${OUTPUT_ROOT}"

for dataset in "${DATASETS[@]}"; do
    data_root="${PROJECT_ROOT}/load/data/${dataset}"
    dataset_exp_root="${OUTPUT_ROOT}/${dataset}"
    metric_root="${OUTPUT_ROOT}/metrics"

    [[ -d "${data_root}" ]] || die "Dataset directory not found: ${data_root}"

    log "Dataset: ${dataset}"

    coarse_command=(
        "${PYTHON_BIN}" "${PROJECT_ROOT}/run_all_coarse_stage.py"
        --data-root "${data_root}"
        --exp-root "${dataset_exp_root}"
        --gpu 0
    )
    refine_command=(
        "${PYTHON_BIN}" "${PROJECT_ROOT}/run_all_0123_refine_stage.py"
        --data-root "${data_root}"
        --exp-root "${dataset_exp_root}"
        --previous-exp-root "${dataset_exp_root}"
        --gpu 0
    )
    texture_command=(
        "${PYTHON_BIN}" "${PROJECT_ROOT}/run_all_pfd0123_texture_stage.py"
        --data-root "${data_root}"
        --exp-root "${dataset_exp_root}"
        --previous-exp-root "${dataset_exp_root}"
        --gpu 0
    )
    if [[ ${#SCENES[@]} -gt 0 ]]; then
        for scene in "${SCENES[@]}"; do
            coarse_command+=(--scene "${scene}")
            refine_command+=(--scene "${scene}")
            texture_command+=(--scene "${scene}")
        done
    fi

    CURRENT_STEP="${dataset} coarse training and test rendering"
    run_command "${coarse_command[@]}"

    CURRENT_STEP="${dataset} refine training and test rendering"
    run_command "${refine_command[@]}"

    CURRENT_STEP="${dataset} texture training and test rendering"
    run_command "${texture_command[@]}"

    CURRENT_STEP="${dataset} five-metric evaluation"
    run_command \
        "${PYTHON_BIN}" "${PROJECT_ROOT}/metric_utils.py" \
        --input-path "${PROJECT_ROOT}/load/data" \
        --pred-path "${dataset_exp_root}/morpheus3d-texture" \
        --datasets "${dataset}" \
        --metrics clip maniqa clipiqa psnr lpips \
        --device 0 \
        --iter 5000 \
        --save-dir "${metric_root}"
done

CURRENT_STEP="complete"
if [[ "${DRY_RUN}" == true ]]; then
    log "Dry run complete; no training or evaluation was executed."
else
    log "Reproduction complete."
    log "Metrics: ${OUTPUT_ROOT}/metrics"
fi
