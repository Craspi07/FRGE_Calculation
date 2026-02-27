"""
Performance and accuracy benchmarks for the FRGE calculation.

Measures:
- Single integration speed
- Pole extraction accuracy
- Memory usage
- Convergence quality
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import tracemalloc
import numpy as np
from core.integrator import run_integration
from core.pole_finder import extract_pole
from config.parameters import HIGGS_MASS_THEORY, HIGGS_MASS_EXP, HIGGS_MASS_ERROR


def benchmark_single_integration(rtol=1e-8, atol=1e-10, n_runs=3):
    """Benchmark a single integration run."""
    times = []
    masses = []

    for i in range(n_runs):
        t0 = time.time()
        result = run_integration(rtol=rtol, atol=atol)
        elapsed = time.time() - t0
        times.append(elapsed)

        if result["success"]:
            pole = extract_pole(result)
            masses.append(pole.get("m_h", float("nan")))

    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "masses": masses,
        "mean_mass": np.nanmean(masses),
        "std_mass": np.nanstd(masses),
        "meets_speed_requirement": np.min(times) < 30.0,
    }


def benchmark_memory_usage():
    """Measure peak memory usage during integration."""
    tracemalloc.start()
    result = run_integration(rtol=1e-8, atol=1e-10)
    pole = extract_pole(result)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / 1024 / 1024
    return {
        "peak_memory_mb": peak_mb,
        "meets_memory_requirement": peak_mb < 2048,  # < 2 GB
        "m_h": pole.get("m_h", float("nan")),
    }


def benchmark_accuracy(n_runs=5):
    """Test accuracy of extracted mass."""
    results = []
    for _ in range(n_runs):
        result = run_integration(rtol=1e-8, atol=1e-10)
        if result["success"]:
            pole = extract_pole(result)
            m_h = pole.get("m_h", float("nan"))
            results.append(m_h)

    results = np.array([m for m in results if not np.isnan(m)])
    if len(results) == 0:
        return {"status": "FAILED", "message": "No valid results"}

    mean_m = np.mean(results)
    std_m = np.std(results)
    deviation_theory = abs(mean_m - HIGGS_MASS_THEORY)
    deviation_exp = abs(mean_m - HIGGS_MASS_EXP)

    return {
        "mean_mass": float(mean_m),
        "std_mass": float(std_m),
        "deviation_from_theory": float(deviation_theory),
        "deviation_from_exp": float(deviation_exp),
        "within_theory_tolerance": deviation_theory < 0.5,
        "within_exp_2sigma": deviation_exp < 2 * HIGGS_MASS_ERROR,
        "reproducibility": float(std_m),
    }


def run_all_benchmarks():
    """Run complete benchmark suite and print report."""
    print("=" * 60)
    print("FRGE HIGGS PROPAGATOR CALCULATION - BENCHMARKS")
    print("=" * 60)
    print()

    print("1. Single Integration Speed (rtol=1e-8, atol=1e-10):")
    speed = benchmark_single_integration(n_runs=2)
    print(f"   Mean time: {speed['mean_time']:.2f}s")
    print(f"   Min time:  {speed['min_time']:.2f}s")
    print(f"   Mean mass: {speed['mean_mass']:.4f} GeV")
    print(f"   Speed OK:  {speed['meets_speed_requirement']}")
    print()

    print("2. Memory Usage:")
    mem = benchmark_memory_usage()
    print(f"   Peak memory: {mem['peak_memory_mb']:.1f} MB")
    print(f"   Memory OK:   {mem['meets_memory_requirement']}")
    print()

    print("3. Accuracy (n=3 runs):")
    acc = benchmark_accuracy(n_runs=3)
    if "message" not in acc:
        print(f"   Mean mass: {acc['mean_mass']:.4f} +/- {acc['std_mass']:.4f} GeV")
        print(f"   Theory deviation: {acc['deviation_from_theory']:.4f} GeV")
        print(f"   Exp deviation:    {acc['deviation_from_exp']:.4f} GeV")
        print(f"   Within theory tol: {acc['within_theory_tolerance']}")
        print(f"   Within 2-sigma:    {acc['within_exp_2sigma']}")
    else:
        print(f"   FAILED: {acc['message']}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    run_all_benchmarks()
