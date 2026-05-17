from pathlib import Path
import site

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parents[1]


def nvidia_wheel_cuda_paths():
    """Return CUDA include/lib paths installed by NVIDIA pip wheels.

    PyTorch CUDA wheels often ship runtime headers and libraries under
    site-packages/nvidia/* instead of a monolithic CUDA toolkit directory.
    Adding these paths lets the extension build against venv-local CUDA 12.x
    toolchains, which is the common layout for vLLM environments.
    """

    include_dirs = []
    library_dirs = []
    candidates = []
    for root in site.getsitepackages() + [site.getusersitepackages()]:
        nvidia_root = Path(root) / "nvidia"
        if nvidia_root.is_dir():
            candidates.append(nvidia_root)

    for nvidia_root in candidates:
        for child in sorted(nvidia_root.iterdir()):
            include_dir = child / "include"
            lib_dir = child / "lib"
            if include_dir.is_dir():
                include_dirs.append(str(include_dir))
            if lib_dir.is_dir():
                library_dirs.append(str(lib_dir))

    return include_dirs, library_dirs


def extension_modules():
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except Exception:
        return [], {}

    nvidia_include_dirs, nvidia_library_dirs = nvidia_wheel_cuda_paths()
    ext = CUDAExtension(
        name="zcutlass_torch._C",
        sources=[
            str(ROOT / "python" / "zcutlass_torch" / "csrc" / "zcutlass_torch.cpp"),
            str(ROOT / "src" / "foundation.cu"),
            str(ROOT / "src" / "gemm.cu"),
        ],
        include_dirs=[str(ROOT / "include"), *nvidia_include_dirs],
        library_dirs=nvidia_library_dirs,
        runtime_library_dirs=nvidia_library_dirs,
        extra_compile_args={
            "cxx": ["-std=c++17"],
            "nvcc": ["-std=c++17", "--expt-relaxed-constexpr", "-gencode=arch=compute_120,code=sm_120"],
        },
    )
    return [ext], {"build_ext": BuildExtension}


ext_modules, cmdclass = extension_modules()

setup(
    name="zcutlass-torch",
    version="0.1.0",
    description="PyTorch overlay proof for zcutlass LLM GEMM buckets",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    entry_points={
        "vllm.general_plugins": [
            "zcutlass_overlay = zcutlass_vllm.plugin:register",
        ],
    },
    python_requires=">=3.10",
)
