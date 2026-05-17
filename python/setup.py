from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parents[1]


def extension_modules():
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except Exception:
        return [], {}

    ext = CUDAExtension(
        name="zcutlass_torch._C",
        sources=[
            str(ROOT / "python" / "zcutlass_torch" / "csrc" / "zcutlass_torch.cpp"),
            str(ROOT / "src" / "foundation.cu"),
            str(ROOT / "src" / "gemm.cu"),
        ],
        include_dirs=[str(ROOT / "include")],
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
    python_requires=">=3.10",
)
