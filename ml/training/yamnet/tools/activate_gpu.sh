# Source this file in a dedicated terminal; it changes only that shell.
_yamnet_gpu_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
_yamnet_gpu_venv="${CAPSTONE_GPU_VENV:-$_yamnet_gpu_root/.venv-gpu}"
if [[ ! -x "$_yamnet_gpu_venv/bin/python" ]]; then
    printf '%s\n' 'Missing GPU virtual environment; installation requires user permission.' >&2
    return 1
fi
export CAPSTONE_GPU_UUID=GPU-10f7f4b0-eee7-9d43-26de-7bbefdbfc7ce
_yamnet_gpu_found="$(nvidia-smi -i "$CAPSTONE_GPU_UUID" --query-gpu=uuid --format=csv,noheader)" || return 1
if [[ "$_yamnet_gpu_found" != "$CAPSTONE_GPU_UUID" ]]; then
    printf '%s\n' 'Selected GPU identity differs; refusing fallback.' >&2
    return 1
fi
source "$_yamnet_gpu_venv/bin/activate" || return 1
_yamnet_gpu_nvidia="$_yamnet_gpu_venv/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$("$VIRTUAL_ENV/bin/python" -c 'import pathlib,sys; print(":".join(str(p) for p in sorted(pathlib.Path(sys.argv[1]).glob("*/lib")) if p.is_dir()))' "$_yamnet_gpu_nvidia")"
export PATH="$_yamnet_gpu_nvidia/cuda_nvcc/bin:$PATH"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$_yamnet_gpu_nvidia/cuda_nvcc"
export CUDA_VISIBLE_DEVICES="$CAPSTONE_GPU_UUID"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CAPSTONE_GPU_CACHE="${CAPSTONE_GPU_CACHE:-$_yamnet_gpu_root/.cache/ubuntu_gpu_cv_s42_v1}"
export CAPSTONE_TF_CACHE="$CAPSTONE_GPU_CACHE/tensorflow"
export CUDA_CACHE_PATH="$CAPSTONE_GPU_CACHE/cuda"
export XDG_CACHE_HOME="$CAPSTONE_GPU_CACHE/xdg"
export PIP_CACHE_DIR="$CAPSTONE_GPU_CACHE/pip"
export KERAS_HOME="$CAPSTONE_TF_CACHE/keras"
export TFHUB_CACHE_DIR="$CAPSTONE_TF_CACHE/tfhub"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TF_USE_LEGACY_KERAS=1
export TF_CPP_MIN_LOG_LEVEL=0
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONNOUSERSITE=1
export PIP_REQUIRE_VIRTUALENV=true
export PIP_DISABLE_PIP_VERSION_CHECK=1
unset _yamnet_gpu_found _yamnet_gpu_nvidia _yamnet_gpu_root _yamnet_gpu_venv
