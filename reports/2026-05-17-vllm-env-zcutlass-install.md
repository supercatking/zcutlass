# vLLM Environment zcutlass Install Proof - 2026-05-17

## Result

`/home/zyz/vllm/.venv` is now usable for the next vLLM overlay development
step. zcutlass was installed into the existing vLLM virtual environment without
modifying the `/home/zyz/vllm` source tree.

Verified facts:

- vLLM: `0.20.0`
- PyTorch: `2.11.0+cu129`
- PyTorch CUDA ABI: `12.9`
- GPU: `NVIDIA GeForce RTX 5080`, capability `(12, 0)`
- zcutlass packages import successfully: `zcutlass_torch`, `zcutlass_vllm`
- vLLM discovers the `zcutlass_overlay` entry point under
  `vllm.general_plugins`
- vLLM plugin loader runs the zcutlass plugin registration path

## Environment Workaround

The first extension build failed because the system default toolkit is CUDA
13.1 while the vLLM environment uses PyTorch `cu129`.

The working build used:

```bash
source /home/zyz/vllm/.venv/bin/activate
python -m pip install 'nvidia-cuda-nvcc-cu12==12.9.*'
python -m pip install 'nvidia-cuda-cccl-cu12==12.9.*'
ln -sf libcudart.so.12 \
  /home/zyz/vllm/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so

cd /home/zyz/zcutlass
export CUDA_HOME=/home/zyz/.triton/nvidia/nvcc/cuda_nvcc-linux-x86_64-12.9.86-archive
export PATH=$CUDA_HOME/bin:/home/zyz/vllm/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export MAX_JOBS=16
python -m pip install -e ./python --no-build-isolation --force-reinstall -v
```

The zcutlass Python build now automatically adds CUDA include/lib directories
from NVIDIA pip wheels under `site-packages/nvidia/*`.

## Validation Commands

```bash
source /home/zyz/vllm/.venv/bin/activate
cd /home/zyz/zcutlass

python tools/check_vllm_env.py --require-vllm --require-zcutlass
python tools/check_torch_overlay.py --require-extension
python tools/check_vllm_plugin.py --require-entry-point --require-vllm
```

Observed results:

```text
PASS: forced_hits=1 policy_misses=1 policy_reasons={'shape_not_target_bucket': 1}
PASS: zcutlass_vllm import/register ok; entry_point_installed=True; vllm=installed
```

Direct vLLM plugin loader proof:

```bash
source /home/zyz/vllm/.venv/bin/activate
cd /home/zyz/zcutlass
python - <<'PY'
import vllm.plugins
import zcutlass_vllm
print("before", zcutlass_vllm.is_registered())
vllm.plugins.load_general_plugins()
print("after", zcutlass_vllm.is_registered())
PY
```

Observed result:

```text
before False
after True
```

## Remaining Gap

This proves that the installed vLLM process can discover and load the zcutlass
overlay plugin. It does not yet prove that vLLM model execution routes a real
Linear/GEMM call to zcutlass or beats stock vLLM. The next milestone is an
explicit vLLM custom model/worker experiment using `ZCutlassVllmLinearAdapter`,
followed by TTFT/TPOT/tokens/s comparison against stock vLLM.
