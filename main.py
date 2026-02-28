#!/usr/bin/env python3
"""
main.py – Entry point for the Higgs Propagator Calculation.

Modes
-----
  python main.py                    Launch GUI
  python main.py --headless         Run single calculation (no GUI)
  python main.py --benchmark        Performance benchmarks
  python main.py --test             Unit tests
  python main.py --scan             2-D parameter scan (θ_i × N_max)
  python main.py --full-scan        Full scan: 3-D grid + 1-D sensitivity for
                                    all 6 physics parameters

Output layout
-------------
  output/                           logs, JSON results, summary text
  output/plots/                     all PNG figures
"""

import sys
import os
import argparse
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.export import (
    OUTPUT_DIR, PLOTS_DIR, setup_output_dirs,
    output_path, save_figure_png, export_results_json,
)
from config.parameters import (
    HIGGS_MASS_TARGET_MIN, HIGGS_MASS_TARGET_MAX,
    HIGGS_MASS_MARGINAL_MIN, HIGGS_MASS_MARGINAL_MAX,
)


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(level=logging.INFO):
    """Write to stdout and output/frge_calculation.log."""
    setup_output_dirs()
    log_path = output_path("frge_calculation.log")
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    return log_path


logger = logging.getLogger(__name__)


# ── Headless ──────────────────────────────────────────────────────────────────

def run_headless(args):
    """Run a single calculation and save all outputs."""
    from core.integrator import run_integration
    from core.pole_finder import extract_pole
    from diagnostics.mixing_analysis import analyze_mixing_term
    from diagnostics.sphere_integral import verify_sphere_enhancement
    from diagnostics.convergence import run_convergence_test
    from utils.export import format_results_summary, export_summary_txt
    from gui.results_panel import build_results_dict
    from gui.plots import create_full_dashboard, save_all_diagnostic_plots
    from config.parameters import (
        THETA_R, THETA_I, DEFAULT_N_MAX, DEFAULT_RTOL, DEFAULT_ATOL,
    )

    theta_r = getattr(args, 'theta_r', THETA_R) or THETA_R
    theta_i = getattr(args, 'theta_i', THETA_I) or THETA_I
    N_max   = getattr(args, 'N_max',  None)     or DEFAULT_N_MAX
    rtol    = getattr(args, 'rtol',   DEFAULT_RTOL)
    atol    = getattr(args, 'atol',   DEFAULT_ATOL)

    logger.info("=" * 64)
    logger.info("HIGGS PROPAGATOR CALCULATION – Reuter Spiral Background")
    logger.info("=" * 64)
    logger.info(f"θ_r={theta_r}  θ_i={theta_i}  N_max={N_max:.4f}")
    logger.info(f"rtol={rtol:.2e}  atol={atol:.2e}")
    logger.info("Running FRGE integration (DOP853, max_step=T_RG/100)…")

    result = run_integration(theta_r=theta_r, theta_i=theta_i, N_max=N_max,
                              rtol=rtol, atol=atol)
    if not result["success"]:
        logger.error(f"Integration FAILED: {result.get('message')}")
        return 1

    stats = result["statistics"]
    logger.info(f"Integration done in {stats['cpu_time']:.2f}s  "
                f"({stats['n_steps']:,} steps, {stats['windings']:.2f} windings)")

    # Pole extraction (Brent + cubic spline, search p²∈[-130²,-120²] GeV²)
    logger.info("Extracting Higgs pole mass…")
    pole_result = extract_pole(result)
    if not pole_result.get("success"):
        logger.warning(f"Pole extraction FAILED: {pole_result.get('message')}")

    m_h = pole_result.get("m_h", float("nan"))

    # Diagnostics
    logger.info("Computing diagnostics…")
    mixing_result = analyze_mixing_term(result)
    sphere_result = verify_sphere_enhancement(n_samples=50000)

    # Convergence test
    logger.info("Running convergence test (progressive tolerance tightening)…")
    conv_result = run_convergence_test(theta_r=theta_r, theta_i=theta_i,
                                       N_max=N_max)
    logger.info(f"Convergence: {conv_result['status']}  "
                f"m_h={conv_result.get('final_mass', float('nan')):.4f} GeV  "
                f"Δm_h={conv_result.get('uncertainty', float('nan')):.2e} GeV")

    # Build unified results dict
    results_dict = build_results_dict(result, pole_result, mixing_result,
                                       sphere_result, convergence_result=conv_result)

    # ── Log summary ───────────────────────────────────────────────────────────
    lines = format_results_summary(results_dict)
    for line in lines:
        logger.info(line)

    # ── Save to output/ ───────────────────────────────────────────────────────
    json_path = export_results_json(results_dict, "results.json")
    logger.info(f"Results JSON     →  {json_path}")
    txt_path  = export_summary_txt(results_dict, "results_summary.txt")
    logger.info(f"Results text     →  {txt_path}")

    # ── Save PNG plots ────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Dashboard (3×3 panel)
    try:
        fig = create_full_dashboard(result, pole_result)
        p = save_figure_png(fig, "dashboard.png")
        logger.info(f"Dashboard PNG    →  {p}")
        plt.close(fig)
    except Exception as exc:
        logger.warning(f"Dashboard plot failed: {exc}")

    # 5 diagnostic plots
    try:
        saved = save_all_diagnostic_plots(result, pole_result,
                                           mixing_result=mixing_result,
                                           sphere_result=sphere_result,
                                           convergence_result=conv_result)
        for p in saved:
            logger.info(f"Diagnostic PNG   →  {p}")
    except Exception as exc:
        logger.warning(f"Diagnostic plots failed: {exc}")

    # Convergence plot (distance to fixed point)
    try:
        from gui.plots import plot_distance_to_fixed_point
        theta_r_p = result.get("params", {}).get("theta_r", theta_r)
        fig, _ = plot_distance_to_fixed_point(
            result["t"], result["y"][0], result["y"][1], theta_r_p
        )
        p = save_figure_png(fig, "convergence_plot.png")
        logger.info(f"Convergence PNG  →  {p}")
        plt.close(fig)
    except Exception as exc:
        logger.warning(f"Convergence plot failed: {exc}")

    # ── Status ────────────────────────────────────────────────────────────────
    if not np.isnan(m_h) and HIGGS_MASS_TARGET_MIN <= m_h <= HIGGS_MASS_TARGET_MAX:
        logger.info("STATUS: [+ CONFIRMED] Theory validated within specification")
        return 0
    elif not np.isnan(m_h) and HIGGS_MASS_MARGINAL_MIN <= m_h <= HIGGS_MASS_MARGINAL_MAX:
        logger.info("STATUS: [~ MARGINAL] Close to target, needs investigation")
        return 0
    else:
        logger.error(f"STATUS: [X FAILED] m_h = {m_h:.3f} GeV outside expected range")
        return 1


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_benchmark():
    from tests.benchmarks import run_all_benchmarks
    run_all_benchmarks()
    return 0


# ── Unit tests ────────────────────────────────────────────────────────────────

def run_tests():
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    return result.returncode


# ── 2-D scan ──────────────────────────────────────────────────────────────────

def run_scan():
    """2-D parameter scan (θ_i × N_max)."""
    from diagnostics.parameter_scan import run_parameter_scan
    from gui.plots import plot_parameter_scan_heatmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logger.info("Running 2-D parameter scan (θ_i × N_max)…")
    result = run_parameter_scan(n_jobs=-1)

    json_path = export_results_json(result, "parameter_scan_results.json")
    sweet = result.get("sweet_spot", {})
    logger.info(f"Sweet spot: θ_i={sweet.get('theta_i','?'):.4f}  "
                f"N_max={sweet.get('N_max','?'):.4f}  "
                f"m_h={sweet.get('m_h','?'):.3f} GeV")
    logger.info(f"Scan JSON        →  {json_path}")

    try:
        fig, ax = plot_parameter_scan_heatmap(
            result["theta_i_arr"], result["N_max_arr"],
            result["mass_grid"], sweet_spot=sweet
        )
        p = save_figure_png(fig, "parameter_scan_2d.png")
        logger.info(f"Scan heatmap PNG →  {p}")
        plt.close(fig)
    except Exception as exc:
        logger.warning(f"Scan heatmap failed: {exc}")

    return 0


# ── Full scan ─────────────────────────────────────────────────────────────────

def run_full_scan():
    """
    Full scan over all 6 physics parameters:
      1-D sensitivity sweeps for: θ_r, θ_i, N_max, A_G, A_Λ, δ_Λ
      3-D grid scan:              θ_r × θ_i × N_max
    Outputs saved to output/ and output/plots/.
    """
    from diagnostics.parameter_scan import run_sensitivity_scan, run_full_grid_scan
    from gui.plots import plot_sensitivity_curves, plot_3d_scan_projections
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logger.info("=" * 64)
    logger.info("FULL PARAMETER SCAN – all 6 physics parameters")
    logger.info("=" * 64)

    # ── 1-D sensitivity sweeps ────────────────────────────────────────────────
    logger.info("Step 1/2 – 1-D sensitivity sweeps (θ_r, θ_i, N_max, A_G, A_Λ, δ_Λ)…")
    sens_result = run_sensitivity_scan(n_jobs=-1)

    json_path = export_results_json(sens_result, "sensitivity_scan_results.json")
    logger.info(f"Sensitivity JSON →  {json_path}")

    try:
        fig, _ = plot_sensitivity_curves(sens_result)
        p = save_figure_png(fig, "sensitivity_curves.png")
        logger.info(f"Sensitivity PNG  →  {p}")
        plt.close(fig)
    except Exception as exc:
        logger.warning(f"Sensitivity curves plot failed: {exc}")

    # Nominal values and per-parameter summary
    for name, data in sens_result.get("sweeps", {}).items():
        masses = data["masses"]
        valid  = masses[~np.isnan(masses)]
        if len(valid) > 0:
            logger.info(
                f"  {name:15s}: m_h range = [{np.min(valid):.3f}, {np.max(valid):.3f}] GeV"
                f"  (std={np.std(valid):.4f} GeV)"
            )

    # ── 3-D grid scan ─────────────────────────────────────────────────────────
    logger.info("Step 2/2 – 3-D grid scan (θ_r × θ_i × N_max)…")
    grid_result = run_full_grid_scan(n_jobs=-1)

    json_path = export_results_json(grid_result, "full_grid_scan_results.json")
    sweet = grid_result.get("sweet_spot", {})
    logger.info(
        f"3-D sweet spot: θ_r={sweet.get('theta_r','?'):.4f}  "
        f"θ_i={sweet.get('theta_i','?'):.4f}  "
        f"N_max={sweet.get('N_max','?'):.4f}  "
        f"m_h={sweet.get('m_h','?'):.3f} GeV"
    )
    logger.info(f"Grid scan JSON   →  {json_path}")

    try:
        fig, _ = plot_3d_scan_projections(grid_result)
        p = save_figure_png(fig, "full_grid_scan_projections.png")
        logger.info(f"Grid scan PNG    →  {p}")
        plt.close(fig)
    except Exception as exc:
        logger.warning(f"Grid scan plot failed: {exc}")

    logger.info("Full scan complete.")
    return 0


# ── GUI ───────────────────────────────────────────────────────────────────────

def launch_gui():
    try:
        import tkinter as tk
        from gui.main_window import MainWindow
        root = tk.Tk()
        MainWindow(root)
        root.mainloop()
        return 0
    except ImportError as exc:
        logger.error(f"GUI requires tkinter: {exc}")
        logger.info("Falling back to headless mode…")
        return run_headless(argparse.Namespace())
    except Exception as exc:
        logger.exception(f"GUI error: {exc}")
        return 1


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Higgs Propagator Calculation – Reuter Spiral Background"
    )
    parser.add_argument("--headless",   action="store_true", help="Single calculation (no GUI)")
    parser.add_argument("--benchmark",  action="store_true", help="Performance benchmarks")
    parser.add_argument("--test",       action="store_true", help="Unit tests")
    parser.add_argument("--scan",       action="store_true", help="2-D parameter scan (θ_i × N_max)")
    parser.add_argument("--full-scan",  action="store_true", help="Full scan: 3-D grid + 1-D sensitivity for all 6 parameters")
    parser.add_argument("--theta-r",    type=float, default=2.714, help="θ_r critical exponent (default: 2.714)")
    parser.add_argument("--theta-i",    type=float, default=2.396, help="θ_i critical exponent (default: 2.396)")
    parser.add_argument("--N-max",      type=float, default=None,  help="Total RG time N_max (default: ln(M_Pl/v) ≈ 39.34)")
    parser.add_argument("--rtol",       type=float, default=1e-8,  help="Relative tolerance (default: 1e-8)")
    parser.add_argument("--atol",       type=float, default=1e-10, help="Absolute tolerance (default: 1e-10)")

    args = parser.parse_args()

    log_path = setup_logging()
    logger.info(f"Log file  : {log_path}")
    logger.info(f"Output dir: {OUTPUT_DIR}")
    logger.info(f"Plots dir : {PLOTS_DIR}")

    if args.benchmark:
        return run_benchmark()
    elif args.test:
        return run_tests()
    elif args.scan:
        return run_scan()
    elif args.full_scan:
        return run_full_scan()
    elif args.headless:
        return run_headless(args)
    else:
        return launch_gui()


if __name__ == "__main__":
    sys.exit(main())
