#!/usr/bin/env python3
"""Benchmark GPU LBM throughput and compare results with a saved GPU baseline."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from controlRB import lbmd2q9


START_SIZE = 300
GRID_GROWTH = 1.25
RAYLEIGH_AT_START_SIZE = 1e7
RAYLEIGH_SIZE_EXPONENT = 3
PRANDTL = 0.71
TAU = 0.5666
WARMUP_STEPS = 2
BENCHMARK_STEPS = 10_000
MAX_TEST_SECONDS = 30.0
DATA_TYPE = "float"
BOTTOM_TEMPERATURE = 2.0
TOP_TEMPERATURE = 1.0
SEED = 42
REFERENCE_PATH = Path(__file__).with_name("gpu_performance_reference.json")


def rayleigh_for_size(size):
    return RAYLEIGH_AT_START_SIZE * (size / START_SIZE) ** RAYLEIGH_SIZE_EXPONENT


def get_gpu_name(device):
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name",
            "--format=csv,noheader",
            "-i",
            str(device),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()[0]


def run_case(size, device):
    name = "gpup{}{}{}".format(os.getpid(), size, time.time_ns())
    rayleigh = rayleigh_for_size(size)
    lbm = lbmd2q9(
        mode="cuda",
        device=device,
        NX=size,
        NY=size,
        NSIM=1,
        TBOT0=BOTTOM_TEMPERATURE,
        TTOP0=TOP_TEMPERATURE,
        Ra=rayleigh,
        Pr=PRANDTL,
        tau=TAU,
        wallprop="WALLED",
        usedtype=DATA_TYPE,
        verbose=False,
        seed=SEED,
        name=name,
    )
    try:
        lbm.allsteps(nsteps=WARMUP_STEPS)
        start = time.perf_counter()
        lbm.allsteps(nsteps=BENCHMARK_STEPS)
        elapsed = time.perf_counter() - start
        mean_squared_velocity = float(
            np.mean(lbm.u[0] ** 2 + lbm.v[0] ** 2)
        )
        return {
            "size": size,
            "rayleigh": rayleigh,
            "iterations_per_second": BENCHMARK_STEPS / elapsed,
            "elapsed_seconds": elapsed,
            "nusselt": float(lbm.calNu()),
            "mean_squared_velocity": mean_squared_velocity,
            "time_steps": int(lbm.t[0]),
        }
    finally:
        lbm.close()


def relative_difference(value, reference):
    if reference == 0:
        return None
    return 100.0 * (value - reference) / abs(reference)


def print_results(results, reference_results):
    print(
        "{:>6} {:>12} {:>12} {:>12} {:>11} {:>12} {:>11} {:>16} {:>12}".format(
            "N",
            "Ra",
            "iter/s",
            "ref iter/s",
            "speed diff %",
            "Nu",
            "Nu diff %",
            "mean(u^2+v^2)",
            "MSE diff %",
        )
    )
    print("-" * 120)
    for result in results:
        reference = reference_results.get(str(result["size"]))
        if reference is None:
            reference_it = speed_diff = reference_nu = nu_diff = None
            reference_msv = msv_diff = None
        else:
            reference_it = reference["iterations_per_second"]
            speed_diff = relative_difference(
                result["iterations_per_second"], reference_it
            )
            reference_nu = reference["nusselt"]
            nu_diff = relative_difference(result["nusselt"], reference_nu)
            reference_msv = reference["mean_squared_velocity"]
            msv_diff = relative_difference(
                result["mean_squared_velocity"], reference_msv
            )

        print(
            "{:6d} {:12.4e} {:12.2f} {:>12} {:>11} {:12.6f} {:>11} {:16.8e} {:>12}".format(
                result["size"],
                result["rayleigh"],
                result["iterations_per_second"],
                "{:.2f}".format(reference_it) if reference_it is not None else "-",
                "{:+.3f}".format(speed_diff) if speed_diff is not None else "-",
                result["nusselt"],
                "{:+.3f}".format(nu_diff) if nu_diff is not None else "-",
                result["mean_squared_velocity"],
                "{:+.3f}".format(msv_diff) if msv_diff is not None else "-",
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", type=int, default=0, help="CUDA device index (default: 0)"
    )
    parser.add_argument(
        "--max-test-seconds",
        type=float,
        default=MAX_TEST_SECONDS,
        help="Stop after the first 10,000-step case taking at least this long",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=REFERENCE_PATH,
        help="JSON baseline path (default: examples/gpu_performance_reference.json)",
    )
    parser.add_argument(
        "--update-reference",
        action="store_true",
        help="Replace an existing baseline with this run's results",
    )
    args = parser.parse_args()
    if not math.isfinite(args.max_test_seconds) or args.max_test_seconds <= 0:
        parser.error("--max-test-seconds must be finite and positive")

    settings = {
        "mode": "cuda",
        "rayleigh_scaling": {
            "value_at_start_size": RAYLEIGH_AT_START_SIZE,
            "start_size": START_SIZE,
            "size_exponent": RAYLEIGH_SIZE_EXPONENT,
        },
        "prandtl": PRANDTL,
        "tau": TAU,
        "bottom_temperature": BOTTOM_TEMPERATURE,
        "top_temperature": TOP_TEMPERATURE,
        "wallprop": "WALLED",
        "dtype": DATA_TYPE,
        "seed": SEED,
        "warmup_steps": WARMUP_STEPS,
        "benchmark_steps": BENCHMARK_STEPS,
        "start_size": START_SIZE,
        "grid_growth": GRID_GROWTH,
    }

    reference = None
    if args.reference.exists() and not args.update_reference:
        reference = json.loads(args.reference.read_text())
        if reference["settings"] != settings:
            raise ValueError(
                "Reference settings differ; use --update-reference to create a new baseline."
            )
        print(
            "Comparing with {} ({})".format(
                args.reference, reference["gpu_name"]
            )
        )
    else:
        print("No baseline loaded; this run will create the reference file.")

    gpu_name = get_gpu_name(args.device)
    print("GPU:", gpu_name)
    print(
        "Timed work per size: {:,} steps; stopping after a case takes {:.1f}s.".format(
            BENCHMARK_STEPS, args.max_test_seconds
        )
    )
    print("Compilation and a {}-step GPU warmup are not timed.".format(WARMUP_STEPS))

    previous_results = reference["results"] if reference else {}
    results = []
    total_test_seconds = 0.0
    size = START_SIZE
    while True:
        result = run_case(size, args.device)
        results.append(result)
        total_test_seconds += result["elapsed_seconds"]
        print(
            "Finished {}x{} in {:.3f}s ({:.2f} iter/s); cumulative timed GPU "
            "runtime {:.3f}s.".format(
                size,
                size,
                result["elapsed_seconds"],
                result["iterations_per_second"],
                total_test_seconds,
            )
        )
        if total_test_seconds >= args.max_test_seconds:
            break
        size = max(size + 1, int(math.ceil(size * GRID_GROWTH)))

    print_results(results, previous_results)

    if reference is None or args.update_reference:
        baseline = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "gpu_name": gpu_name,
            "settings": settings,
            "results": {
                str(result["size"]): result
                for result in results
            },
        }
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        args.reference.write_text(json.dumps(baseline, indent=2) + "\n")
        print("Created GPU reference:", args.reference)
    elif gpu_name != reference["gpu_name"]:
        print(
            "Warning: baseline GPU is {}; current GPU is {}.".format(
                reference["gpu_name"], gpu_name
            )
        )
    unmatched = [
        result["size"]
        for result in results
        if str(result["size"]) not in previous_results
    ]
    if reference is not None and unmatched:
        print(
            "No saved reference for grid size(s) {}. Use --update-reference "
            "to replace the baseline.".format(", ".join(map(str, unmatched)))
        )


if __name__ == "__main__":
    main()
