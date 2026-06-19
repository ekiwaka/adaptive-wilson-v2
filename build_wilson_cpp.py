from __future__ import annotations
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "cpp" / "wilson_cpu.cpp"
PKG = ROOT / "src" / "gnn_graph_sampling"


def build() -> Path:
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    output = PKG / f"libwilson_cpu{suffix}"
    command = [
        "c++",
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        str(SRC),
        "-o",
        str(output),
    ]
    print("Building:", " ".join(command))
    subprocess.run(command, check=True)
    print("Built:", output)
    return output


if __name__ == "__main__":
    build()
