# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

from pathlib import Path

from setuptools import setup

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "python" / "fmha_sm100" / "csrc" / "kvouter"


def _build_extension():
    """Build the KV-outer CUDA extension when build deps are available."""
    import nvidia_cutlass_dsl
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

    def cute_dsl_paths() -> tuple[Path, Path]:
        for base in map(Path, nvidia_cutlass_dsl.__path__):
            include_dir = base / "include"
            library_dir = base / "lib"
            if (
                (include_dir / "CuteDSLRuntime.h").is_file()
                and (library_dir / "libcute_dsl_runtime.so").is_file()
            ):
                return include_dir, library_dir
        raise RuntimeError(
            "nvidia-cutlass-dsl does not provide CuteDSLRuntime.h and "
            "libcute_dsl_runtime.so; install its CUDA 13 runtime extra"
        )

    def cuda_driver_library_dir() -> Path:
        if CUDA_HOME is None:
            raise RuntimeError("CUDA_HOME is not set; a CUDA 13 toolkit is required")
        cuda_home = Path(CUDA_HOME)
        for candidate in (
            cuda_home / "lib64" / "stubs",
            cuda_home / "lib" / "stubs",
            cuda_home / "targets" / "x86_64-linux" / "lib" / "stubs",
        ):
            if (candidate / "libcuda.so").is_file():
                return candidate
        raise RuntimeError(f"could not find the CUDA driver stub under {cuda_home}")

    cute_include, cute_library = cute_dsl_paths()
    cuda_driver_library = cuda_driver_library_dir()
    return CUDAExtension(
        name="fmha_sm100._C",
        sources=[
            str((CSRC / "bindings.cpp").relative_to(ROOT)),
            str((CSRC / "cute_sparse_kvouter.cpp").relative_to(ROOT)),
        ],
        include_dirs=[str(CSRC), str(cute_include)],
        library_dirs=[str(cuda_driver_library), str(cute_library)],
        libraries=["cuda", "cute_dsl_runtime"],
        extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
    )


ext_modules = []
cmdclass = {}
try:
    ext_modules = [_build_extension()]
    from torch.utils.cpp_extension import BuildExtension

    cmdclass = {"build_ext": BuildExtension.with_options(use_ninja=True)}
except Exception:
    # PEP 517 metadata hooks import setup.py before CUDA/CuTe build deps exist.
    ext_modules = []
    cmdclass = {}

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
)
