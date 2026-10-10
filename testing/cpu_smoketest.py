#!/usr/bin/env python3
"""Build the CPU LBM library and run a small end-to-end simulation."""

import importlib.util
import os
from pathlib import Path
import subprocess

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
    library_name = "cpusmoke{}".format(os.getpid())

    try:
        lbmd2q9 = load_lbm_class(repo_root)
        lbm = lbmd2q9(
            mode="cpu",
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

        print("CPU smoke test passed")
    finally:
        subprocess.run(
            ["make", "clean", "name={}".format(library_name)],
            cwd=str(repo_root / "d2q9-cpu"),
            check=True,
        )


if __name__ == "__main__":
    main()
