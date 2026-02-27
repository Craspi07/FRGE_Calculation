#!/usr/bin/env python3
"""
main.py - Entry point for the Higgs Propagator Calculation.

Launches the GUI application or runs in headless/CLI mode.

Usage:
    python main.py                  # Launch GUI
    python main.py --headless       # Run headless calculation
    python main.py --benchmark      # Run performance benchmarks
    python main.py --test           # Run unit tests
    python main.py --scan           # Run parameter scan
"""

import sys
import os
import argparse
import numpy as np

# Ensure package root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_headless(args):
    """Run calculation in headless mode (no GUI)."""
    from core.integrator import run_integration
    from core.pole_finder import extract_pole
    from core.spiral import count_spiral_windings
    from diagnostics.mixing_analysis import analyze_mixing_term
    from diagnostics.sphere_integral import verify_sphere_enhancement
    from utils.export import format_results_summary
    from config.parameters import (
        THETA_R, THETA_I, DEFAULT_N_MAX, DEFAULT_RTOL, DEFAULT_ATOL,
        HIGGS_MASS_THEORY, HIGGS_MASS_EXP
    )

    theta_r = getattr(args, 'theta_r', THETA_R) or THETA_R
    theta_i = getattr(args, 'theta_i', THETA_I) or THETA_I
    N_max   = getattr(args, 'N_max', None) or DEFAULT_N_MAX
    rtol    = getattr(args, 'rtol', DEFAULT_RTOL)
    atol    = getattr(args, 'atol', DEFAULT_ATOL)

    print("=" * 60)
    print("HIGGS PROPAGATOR CALCULATION - Reuter Spiral Background")
    print("=" * 60)
    print(f"Parameters: theta_r={theta_r}, theta_i={theta_i}, N_max={N_max:.4f}")
    print(f"Tolerances: rtol={rtol:.2e}, atol={atol:.2e}")
    print()
    print("Running FRGE integration...")

    # Integration
    result = run_integration(
        theta_r=theta_r, theta_i=theta_i, N_max=N_max,
        rtol=rtol, atol=atol
    )

    if not result["success"]:
        print(f"Integration FAILED: {result.get('message')}")
        return 1

    stats = result["statistics"]
    print(f"Integration complete in {stats['cpu_time']:.1f}s")
    print(f"Steps: {stats['n_steps']:,} | Windings: {stats['windings']:.2f}")
    print()

    # Pole extraction
    print("Extracting Higgs pole mass...")
    pole_result = extract_pole(result)

    if not pole_result.get("success"):
        print(f"Pole extraction FAILED: {pole_result.get('message')}")
        print("Trying with fallback parameters...")

    m_h = pole_result.get("m_h", float("nan"))
    m_h_unc = pole_result.get("m_h_uncertainty", float("nan"))

    # Diagnostics
    print("Computing diagnostics...")
    mixing_result = analyze_mixing_term(result)
    sphere_result = verify_sphere_enhancement(n_samples=50000)

    # Build and display results
    from gui.results_panel import build_results_dict
    results_dict = build_results_dict(result, pole_result, mixing_result, sphere_result)
    lines = format_results_summary(results_dict)
    print()
    print("\n".join(lines))
    print()

    # Validation
    theory_agreement = abs(m_h - HIGGS_MASS_THEORY) if not np.isnan(m_h) else float("nan")
    exp_agreement = abs(m_h - HIGGS_MASS_EXP) if not np.isnan(m_h) else float("nan")

    if not np.isnan(m_h) and 125.0 <= m_h <= 125.4:
        print("STATUS: [+ CONFIRMED] Theory validated within precision")
        return 0
    elif not np.isnan(m_h) and 124.0 <= m_h <= 126.5:
        print("STATUS: [~ MARGINAL] Close to target, needs investigation")
        return 0
    else:
        print(f"STATUS: [X FAILED] m_h = {m_h:.3f} GeV outside expected range")
        return 1


def run_benchmark():
    """Run performance benchmarks."""
    from tests.benchmarks import run_all_benchmarks
    run_all_benchmarks()
    return 0


def run_tests():
    """Run the unit test suite."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    return result.returncode


def run_scan():
    """Run parameter scan and save results."""
    from diagnostics.parameter_scan import run_parameter_scan
    from utils.export import export_results_json
    print("Running parameter scan (this may take several minutes)...")
    result = run_parameter_scan(n_jobs=-1)
    export_results_json(result, "parameter_scan_results.json")
    sweet = result.get("sweet_spot", {})
    print(f"Sweet spot: theta_i={sweet.get('theta_i','?'):.4f}, "
          f"N_max={sweet.get('N_max','?'):.4f}, m_h={sweet.get('m_h','?'):.3f} GeV")
    print("Results saved to parameter_scan_results.json")
    return 0


def launch_gui():
    """Launch the tkinter GUI application."""
    try:
        import tkinter as tk
        from gui.main_window import MainWindow
        root = tk.Tk()
        app = MainWindow(root)
        root.mainloop()
        return 0
    except ImportError as e:
        print(f"GUI requires tkinter: {e}")
        print("Falling back to headless mode...")
        return run_headless(argparse.Namespace())
    except Exception as e:
        print(f"GUI error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Higgs Propagator Calculation in Reuter Spiral Background"
    )
    parser.add_argument("--headless", action="store_true",
                        help="Run headless (no GUI)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run performance benchmarks")
    parser.add_argument("--test", action="store_true",
                        help="Run unit tests")
    parser.add_argument("--scan", action="store_true",
                        help="Run parameter scan")
    parser.add_argument("--theta-r", type=float, default=2.714,
                        help="theta_r critical exponent (default: 2.714)")
    parser.add_argument("--theta-i", type=float, default=2.396,
                        help="theta_i critical exponent (default: 2.396)")
    parser.add_argument("--N-max", type=float, default=None,
                        help="Total RG time N_max (default: ln(M_Pl/v) ~ 38.44)")
    parser.add_argument("--rtol", type=float, default=1e-8,
                        help="Relative tolerance (default: 1e-8)")
    parser.add_argument("--atol", type=float, default=1e-10,
                        help="Absolute tolerance (default: 1e-10)")

    args = parser.parse_args()

    if args.benchmark:
        return run_benchmark()
    elif args.test:
        return run_tests()
    elif args.scan:
        return run_scan()
    elif args.headless:
        return run_headless(args)
    else:
        return launch_gui()


if __name__ == "__main__":
    sys.exit(main())
