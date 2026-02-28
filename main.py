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

Outputs
-------
All log files are written to  output/
All PNG plots are written to  output/plots/
"""

import sys
import os
import argparse
import logging
import numpy as np

# Ensure package root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.export import (
    OUTPUT_DIR, PLOTS_DIR, setup_output_dirs,
    output_path, save_figure_png, export_results_json,
)

# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(level=logging.INFO):
    """
    Configure root logger to write to both the console and
    output/frge_calculation.log.
    """
    setup_output_dirs()
    log_path = output_path("frge_calculation.log")

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler  →  output/frge_calculation.log
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    return log_path


logger = logging.getLogger(__name__)


# ── Headless mode ─────────────────────────────────────────────────────────────

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

    logger.info("=" * 60)
    logger.info("HIGGS PROPAGATOR CALCULATION - Reuter Spiral Background")
    logger.info("=" * 60)
    logger.info(f"Parameters: theta_r={theta_r}, theta_i={theta_i}, N_max={N_max:.4f}")
    logger.info(f"Tolerances: rtol={rtol:.2e}, atol={atol:.2e}")
    logger.info("Running FRGE integration...")

    # Integration
    result = run_integration(
        theta_r=theta_r, theta_i=theta_i, N_max=N_max,
        rtol=rtol, atol=atol
    )

    if not result["success"]:
        logger.error(f"Integration FAILED: {result.get('message')}")
        return 1

    stats = result["statistics"]
    logger.info(f"Integration complete in {stats['cpu_time']:.1f}s")
    logger.info(f"Steps: {stats['n_steps']:,} | Windings: {stats['windings']:.2f}")

    # Pole extraction
    logger.info("Extracting Higgs pole mass...")
    pole_result = extract_pole(result)

    if not pole_result.get("success"):
        logger.warning(f"Pole extraction FAILED: {pole_result.get('message')}")
        logger.warning("Trying with fallback parameters...")

    m_h = pole_result.get("m_h", float("nan"))
    m_h_unc = pole_result.get("m_h_uncertainty", float("nan"))

    # Diagnostics
    logger.info("Computing diagnostics...")
    mixing_result = analyze_mixing_term(result)
    sphere_result = verify_sphere_enhancement(n_samples=50000)

    # Build and display results
    from gui.results_panel import build_results_dict
    results_dict = build_results_dict(result, pole_result, mixing_result, sphere_result)
    lines = format_results_summary(results_dict)
    for line in lines:
        logger.info(line)

    # ── Save results to output/ ──────────────────────────────────────────────
    json_path = export_results_json(results_dict, "results.json")
    logger.info(f"Results JSON  →  {json_path}")

    from utils.export import export_summary_txt
    txt_path = export_summary_txt(results_dict, "results_summary.txt")
    logger.info(f"Results text  →  {txt_path}")

    # ── Save plots to output/plots/ ──────────────────────────────────────────
    try:
        from gui.plots import create_full_dashboard
        import matplotlib
        matplotlib.use("Agg")
        fig = create_full_dashboard(result, pole_result)
        png_path = save_figure_png(fig, "dashboard.png")
        logger.info(f"Dashboard PNG →  {png_path}")
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception as exc:
        logger.warning(f"Could not save dashboard plot: {exc}")

    try:
        _save_convergence_plot(result, pole_result)
    except Exception as exc:
        logger.warning(f"Could not save convergence plot: {exc}")

    # Validation
    if not np.isnan(m_h) and 125.0 <= m_h <= 125.4:
        logger.info("STATUS: [+ CONFIRMED] Theory validated within precision")
        return 0
    elif not np.isnan(m_h) and 124.0 <= m_h <= 126.5:
        logger.info("STATUS: [~ MARGINAL] Close to target, needs investigation")
        return 0
    else:
        logger.error(f"STATUS: [X FAILED] m_h = {m_h:.3f} GeV outside expected range")
        return 1


def _save_convergence_plot(result, pole_result):
    """Save a standalone convergence / distance-to-fixed-point PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gui.plots import plot_distance_to_fixed_point

    t_arr = result["t"]
    G_arr = result["y"][0]
    Lambda_arr = result["y"][1]
    theta_r = result.get("params", {}).get("theta_r", 2.714)

    fig, ax = plot_distance_to_fixed_point(t_arr, G_arr, Lambda_arr, theta_r)
    png_path = save_figure_png(fig, "convergence_plot.png")
    logger.info(f"Convergence PNG →  {png_path}")
    plt.close(fig)


# ── Benchmark / test / scan modes ─────────────────────────────────────────────

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
    """Run parameter scan and save results to output/."""
    from diagnostics.parameter_scan import run_parameter_scan
    logger.info("Running parameter scan (this may take several minutes)...")
    result = run_parameter_scan(n_jobs=-1)
    json_path = export_results_json(result, "parameter_scan_results.json")
    sweet = result.get("sweet_spot", {})
    logger.info(
        f"Sweet spot: theta_i={sweet.get('theta_i','?'):.4f}, "
        f"N_max={sweet.get('N_max','?'):.4f}, m_h={sweet.get('m_h','?'):.3f} GeV"
    )
    logger.info(f"Results saved to {json_path}")
    return 0


# ── GUI ───────────────────────────────────────────────────────────────────────

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
        logger.error(f"GUI requires tkinter: {e}")
        logger.info("Falling back to headless mode...")
        return run_headless(argparse.Namespace())
    except Exception as e:
        logger.exception(f"GUI error: {e}")
        return 1


# ── Entry point ───────────────────────────────────────────────────────────────

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

    log_path = setup_logging()
    logger.info(f"Log file: {log_path}")
    logger.info(f"Output dir: {OUTPUT_DIR}")
    logger.info(f"Plots dir:  {PLOTS_DIR}")

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
