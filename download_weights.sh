#!/usr/bin/env bash

# Download the minimal pretrained weights required by the three training stages
# and the five evaluation metrics. Run this file from any working directory:
#
#   ./download_weights.sh
#

set -Eeuo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

readonly ZERO123_REPO="bennyguo/zero123-xl-diffusers"
readonly ZERO123_REVISION="ae385edfaf74c677408f96bd04b513483bd326ac"
readonly PFD_REPO="shi-labs/prompt-free-diffusion"
readonly PFD_REVISION="b9b8a9079e2457f4c5af77d7e3261e03a5747e46"
readonly CLIP_REPO="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
readonly CLIP_REVISION="8c7a3583335de4dba1b07182dbf81c75137ce67b"
readonly TIMM_REPO="timm/vit_base_patch8_224.augreg2_in21k_ft_in1k"
readonly TIMM_REVISION="9c48fc9fd87ab0a53f3368f1f7bccb20fca62aad"

readonly DIFFUSION_ROOT="${PROJECT_ROOT}/diffusion_ckpt"
readonly ZERO123_DIR="${DIFFUSION_ROOT}/zero123-xl-diffusers-base"
readonly PFD_STORE="${PROJECT_ROOT}/weights/prompt-free-diffusion"
readonly PFD_PRETRAINED_DIR="${PFD_STORE}/pretrained"
readonly PFD_BASE_LINK="${DIFFUSION_ROOT}/prompt-free-diffusion-base"
readonly PFD_LORA_LINK="${DIFFUSION_ROOT}/prompt-free-diffusion"
readonly PFD_EXTERN_LINK="${PROJECT_ROOT}/extern/pfd/pretrained"
readonly CLIP_ROOT="${PROJECT_ROOT}/clip_ckpt"
readonly CLIP_DIR="${CLIP_ROOT}/CLIP-ViT-bigG-14-laion2B-39B-b160k"
readonly METRIC_DIR="${PROJECT_ROOT}/metric_ckpt"
readonly TIMM_DIR="${METRIC_DIR}/vit_base_patch8_224.augreg2_in21k_ft_in1k"

log() {
    printf '[download_weights] %s\n' "$*"
}

die() {
    printf '[download_weights] ERROR: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

select_huggingface_cli() {
    if command -v hf >/dev/null 2>&1; then
        HF_DOWNLOAD_COMMAND=(hf download)
        HF_LEGACY_CLI=false
    elif command -v huggingface-cli >/dev/null 2>&1; then
        HF_DOWNLOAD_COMMAND=(huggingface-cli download)
        HF_LEGACY_CLI=true
    else
        die "Install huggingface_hub first: python -m pip install -U huggingface_hub"
    fi
}

check_download_directory() {
    local path="$1"

    if [[ -L "${path}" ]]; then
        die "Refusing to download through symbolic link: ${path}"
    fi
    if [[ -e "${path}" && ! -d "${path}" ]]; then
        die "Expected a directory but found another file type: ${path}"
    fi
}

check_link_slot() {
    local link_path="$1"
    local expected_target="$2"

    if [[ -L "${link_path}" ]]; then
        local current_target
        current_target="$(readlink "${link_path}")"
        [[ "${current_target}" == "${expected_target}" ]] || \
            die "Existing link has an unexpected target: ${link_path} -> ${current_target}"
    elif [[ -e "${link_path}" ]]; then
        die "Refusing to replace existing path: ${link_path}"
    fi
}

download_huggingface_files() {
    local repo_id="$1"
    local revision="$2"
    local destination="$3"
    shift 3

    local command_args=(
        "${HF_DOWNLOAD_COMMAND[@]}"
        "${repo_id}"
        "$@"
        --revision "${revision}"
        --local-dir "${destination}"
    )
    if [[ "${HF_LEGACY_CLI}" == true ]]; then
        command_args+=(--local-dir-use-symlinks False --resume-download)
    fi

    log "Downloading ${repo_id}@${revision}"
    "${command_args[@]}"
}

download_url() {
    local url="$1"
    local destination="$2"

    if [[ -s "${destination}" ]]; then
        log "Already present: ${destination#"${PROJECT_ROOT}/"}"
        return
    fi

    mkdir -p "$(dirname "${destination}")"
    local temporary_file
    temporary_file="$(mktemp "${destination}.part.XXXXXX")"

    log "Downloading ${url}"
    if ! curl --fail --location --retry 3 --retry-delay 2 \
        --output "${temporary_file}" "${url}"; then
        rm -f "${temporary_file}"
        die "Download failed: ${url}"
    fi
    mv -f "${temporary_file}" "${destination}"
}

ensure_symlink() {
    local link_path="$1"
    local relative_target="$2"

    if [[ -L "${link_path}" ]]; then
        log "Already linked: ${link_path#"${PROJECT_ROOT}/"}"
        return
    fi

    mkdir -p "$(dirname "${link_path}")"
    ln -s "${relative_target}" "${link_path}"
    log "Linked ${link_path#"${PROJECT_ROOT}/"} -> ${relative_target}"
}

require_file() {
    [[ -f "$1" ]] || die "Required file is missing after download: $1"
}

main() {
    cd "${PROJECT_ROOT}"
    umask 022

    require_command curl
    select_huggingface_cli

    check_download_directory "${DIFFUSION_ROOT}"
    check_download_directory "${ZERO123_DIR}"
    check_download_directory "${PROJECT_ROOT}/weights"
    check_download_directory "${PFD_STORE}"
    check_download_directory "${PFD_PRETRAINED_DIR}"
    check_download_directory "${CLIP_ROOT}"
    check_download_directory "${CLIP_DIR}"
    check_download_directory "${METRIC_DIR}"
    check_download_directory "${TIMM_DIR}"
    check_link_slot "${PFD_BASE_LINK}" "../weights/prompt-free-diffusion/pretrained"
    check_link_slot "${PFD_LORA_LINK}" "../weights/prompt-free-diffusion/pretrained"
    check_link_slot "${PFD_EXTERN_LINK}" "../../weights/prompt-free-diffusion/pretrained"

    mkdir -p "${DIFFUSION_ROOT}" "${PFD_STORE}" "${CLIP_ROOT}" "${METRIC_DIR}"

    download_huggingface_files \
        "${ZERO123_REPO}" "${ZERO123_REVISION}" "${ZERO123_DIR}" \
        model_index.json \
        feature_extractor/preprocessor_config.json \
        scheduler/scheduler_config.json \
        clip_camera_projection/config.json \
        clip_camera_projection/diffusion_pytorch_model.fp16.safetensors \
        image_encoder/config.json \
        image_encoder/model.fp16.safetensors \
        unet/config.json \
        unet/diffusion_pytorch_model.fp16.safetensors \
        vae/config.json \
        vae/diffusion_pytorch_model.fp16.safetensors

    download_huggingface_files \
        "${PFD_REPO}" "${PFD_REVISION}" "${PFD_STORE}" \
        pretrained/controlnet/control_sd15_canny_slimmed.safetensors \
        pretrained/pfd/diffuser/SD-v1-5.safetensors \
        pretrained/pfd/seecoder/seecoder-v1-0.safetensors \
        pretrained/pfd/vae/sd-v2-0-base-autokl.pth

    require_file "${PFD_PRETRAINED_DIR}/controlnet/control_sd15_canny_slimmed.safetensors"
    require_file "${PFD_PRETRAINED_DIR}/pfd/diffuser/SD-v1-5.safetensors"
    require_file "${PFD_PRETRAINED_DIR}/pfd/seecoder/seecoder-v1-0.safetensors"
    require_file "${PFD_PRETRAINED_DIR}/pfd/vae/sd-v2-0-base-autokl.pth"

    # Keep the two configuration paths distinct so PFD creates separate frozen
    # and trainable-LoRA pipelines, while all three paths share one disk copy.
    ensure_symlink "${PFD_BASE_LINK}" "../weights/prompt-free-diffusion/pretrained"
    ensure_symlink "${PFD_LORA_LINK}" "../weights/prompt-free-diffusion/pretrained"
    ensure_symlink "${PFD_EXTERN_LINK}" "../../weights/prompt-free-diffusion/pretrained"

    download_huggingface_files \
        "${CLIP_REPO}" "${CLIP_REVISION}" "${CLIP_DIR}" \
        config.json \
        pytorch_model.bin.index.json \
        pytorch_model-00001-of-00002.bin \
        pytorch_model-00002-of-00002.bin

    download_huggingface_files \
        "${TIMM_REPO}" "${TIMM_REVISION}" "${TIMM_DIR}" \
        model.safetensors

    download_url \
        "https://github.com/IIGROUP/MANIQA/releases/download/Koniq10k/ckpt_koniq10k.pt" \
        "${METRIC_DIR}/ckpt_koniq10k.pt"
    download_url \
        "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt" \
        "${METRIC_DIR}/RN50.pt"

    # lpips==0.1.4 bundles its linear calibration weights, but torchvision's
    # AlexNet backbone normally downloads on first use. Cache it in advance.
    local torch_cache_root
    if [[ -n "${TORCH_HOME:-}" ]]; then
        torch_cache_root="${TORCH_HOME}/hub"
    elif [[ -n "${XDG_CACHE_HOME:-}" ]]; then
        torch_cache_root="${XDG_CACHE_HOME}/torch/hub"
    else
        [[ -n "${HOME:-}" ]] || die "HOME is unset; cannot determine the PyTorch cache directory"
        torch_cache_root="${HOME}/.cache/torch/hub"
    fi
    download_url \
        "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth" \
        "${torch_cache_root}/checkpoints/alexnet-owt-7be5be79.pth"

    local required_files=(
        "${ZERO123_DIR}/model_index.json"
        "${ZERO123_DIR}/feature_extractor/preprocessor_config.json"
        "${ZERO123_DIR}/scheduler/scheduler_config.json"
        "${ZERO123_DIR}/clip_camera_projection/config.json"
        "${ZERO123_DIR}/clip_camera_projection/diffusion_pytorch_model.fp16.safetensors"
        "${ZERO123_DIR}/image_encoder/config.json"
        "${ZERO123_DIR}/image_encoder/model.fp16.safetensors"
        "${ZERO123_DIR}/unet/config.json"
        "${ZERO123_DIR}/unet/diffusion_pytorch_model.fp16.safetensors"
        "${ZERO123_DIR}/vae/config.json"
        "${ZERO123_DIR}/vae/diffusion_pytorch_model.fp16.safetensors"
        "${PFD_BASE_LINK}/controlnet/control_sd15_canny_slimmed.safetensors"
        "${PFD_BASE_LINK}/pfd/seecoder/seecoder-v1-0.safetensors"
        "${PFD_LORA_LINK}/pfd/diffuser/SD-v1-5.safetensors"
        "${PFD_EXTERN_LINK}/pfd/vae/sd-v2-0-base-autokl.pth"
        "${CLIP_DIR}/config.json"
        "${CLIP_DIR}/pytorch_model.bin.index.json"
        "${CLIP_DIR}/pytorch_model-00001-of-00002.bin"
        "${CLIP_DIR}/pytorch_model-00002-of-00002.bin"
        "${TIMM_DIR}/model.safetensors"
        "${METRIC_DIR}/ckpt_koniq10k.pt"
        "${METRIC_DIR}/RN50.pt"
        "${torch_cache_root}/checkpoints/alexnet-owt-7be5be79.pth"
    )
    local required_file
    for required_file in "${required_files[@]}"; do
        require_file "${required_file}"
    done

    log "All required training and evaluation weights are ready."
}

main "$@"
