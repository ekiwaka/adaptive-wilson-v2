from __future__ import annotations
from pathlib import Path
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension, CUDA_HOME

ROOT = Path(__file__).resolve().parent
CPP_DIR = ROOT / "cpp"

sources = [
    str(CPP_DIR / "wilson_extension.cpp"),
    str(CPP_DIR / "wilson_ops_cpu.cpp"),
]
use_cuda = CUDA_HOME is not None and torch.version.cuda is not None
extra_compile_args: dict[str, list[str]] = {"cxx": ["-O3", "-std=c++17"]}
define_macros: list[tuple[str, None]] = []
extension_class = CppExtension

if use_cuda:
    extension_class = CUDAExtension
    sources.append(str(CPP_DIR / "wilson_ops_cuda.cu"))
    define_macros.append(("WITH_CUDA", None))
    extra_compile_args["nvcc"] = ["-O3", "--use_fast_math"]

setup(
    name="gnn_graph_sampling_wilson_ext",
    ext_modules=[
        extension_class(
            name="gnn_graph_sampling._wilson_ext",
            sources=sources,
            include_dirs=[str(CPP_DIR)],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    package_dir={"": "src"},
)
