#!/usr/bin/env python3
"""Build the CUDA LBM library and run a small end-to-end GPU simulation."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def load_lbm_class(repo_root):
    wrapper_path = repo_root / "controlRB" / "LBMd2q9ctypes.py"
    spec = importlib.util.spec_from_file_location("lbm_wrapper", str(wrapper_path))
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load the LBM Python wrapper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.lbmd2q9


def main():
    repo_root = Path(__file__).resolve().parents[1]
    library_name = "gpusmoke{}".format(os.getpid())
    cuda_path = Path(os.environ.get("CUDA_PATH", "/usr/local/cuda")).expanduser()
    nvcc = cuda_path / "bin" / "nvcc"

    if not nvcc.is_file():
        print(
            "CUDA compiler not found at {}. Install the CUDA Toolkit in WSL "
            "or set CUDA_PATH to its installation directory.".format(nvcc),
            file=sys.stderr,
        )
        return 1

    try:
        subprocess.run(["nvidia-smi", "-L"], check=True)
    except FileNotFoundError:
        print("nvidia-smi not found; cannot verify an NVIDIA GPU is available.", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:
        print("nvidia-smi could not query the GPU (exit {}).".format(error.returncode), file=sys.stderr)
        return 1

    lbm = None
    try:
        lbmd2q9 = load_lbm_class(repo_root)
        lbm = lbmd2q9(
            mode="cuda",
            device=0,
            NX=16,
            NY=16,
            NSIM=1,
            Ra=1e4,
            verbose=True,
            name=library_name,
        )

        print("Initial time:", lbm.t.tolist())
        lbm.allsteps(nsteps=20)

        print("Final time:", lbm.t.tolist())
        print(
            "Temperature range:",
            float(lbm.temp.min()),
            float(lbm.temp.max()),
        )
        print(
            "Maximum absolute velocity:",
            float(max(np.abs(lbm.u).max(), np.abs(lbm.v).max())),
        )

        assert lbm.t.tolist() == [20], "Expected the time counter to advance by 20"
        assert lbm.temp.shape == (1, 16, 16), "Unexpected temperature array shape"
        for field_name in ("temp", "u", "v", "rho"):
            assert np.isfinite(getattr(lbm, field_name)).all(), (
                "{} contains non-finite values".format(field_name)
            )

        print("GPU smoke test passed")
        return 0
    finally:
        try:
            if lbm is not None and not lbm.closed:
                lbm.close()
        finally:
            cuda_dir = repo_root / "d2q9-cuda"
            for filename in (
                "rlbm{}".format(library_name),
                "rlbm{}.o".format(library_name),
                "rlbm_gpu{}.o".format(library_name),
                "rlbm{}.so".format(library_name),
                "lbm{}.so".format(library_name),
                "rlbmso{}.o".format(library_name),
                "rlbm_gpuso{}.o".format(library_name),
                "test{}.out".format(library_name),
            ):
                artifact = cuda_dir / filename
                if artifact.exists():
                    artifact.unlink()


if __name__ == "__main__":
    sys.exit(main())
