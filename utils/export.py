"""Data export and import functions."""

import json
import os
import numpy as np

# ── Output directory layout ──────────────────────────────────────────────────
#   output/           ← logs, JSON results, summary .txt files
#   output/plots/     ← all PNG figures
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR    = os.path.join(_PROJECT_ROOT, "output")
PLOTS_DIR     = os.path.join(OUTPUT_DIR, "plots")


def setup_output_dirs():
    """Create output/ and output/plots/ if they don't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,  exist_ok=True)


def output_path(filename):
    """Return an absolute path inside output/."""
    setup_output_dirs()
    return os.path.join(OUTPUT_DIR, filename)


def plots_path(filename):
    """Return an absolute path inside output/plots/."""
    setup_output_dirs()
    return os.path.join(PLOTS_DIR, filename)


def save_figure_png(fig, filename):
    """
    Save a matplotlib Figure as a PNG inside output/plots/.

    Parameters
    ----------
    fig      : matplotlib.figure.Figure
    filename : str   e.g. "dashboard.png"  (basename only, no directory)

    Returns
    -------
    str  – the full path where the file was written
    """
    path = plots_path(filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def export_results_json(results, filepath=None):
    """Export calculation results to JSON file.

    If *filepath* is a bare filename (no directory component) it is placed
    inside ``output/``.  Pass an absolute/relative path to override.
    """
    if filepath is None:
        filepath = output_path("results.json")
    elif not os.path.dirname(filepath):
        filepath = output_path(filepath)

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    def _recursive_convert(d):
        if isinstance(d, dict):
            return {k: _recursive_convert(v) for k, v in d.items()}
        if isinstance(d, (list, tuple)):
            return [_recursive_convert(v) for v in d]
        return _convert(d)

    serializable = _recursive_convert(results)
    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)
    return filepath


def export_trajectory_npz(t_arr, y_arr, filepath=None):
    """Export trajectory data to numpy .npz file inside output/."""
    if filepath is None:
        filepath = output_path("trajectory")
    elif not os.path.dirname(filepath):
        filepath = output_path(filepath)
    np.savez_compressed(filepath, t=t_arr, y=y_arr)
    return filepath + ".npz"


def load_results_json(filepath):
    """Load results from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def export_summary_txt(results, filepath=None):
    """Export formatted summary text to file inside output/."""
    if filepath is None:
        filepath = output_path("results_summary.txt")
    elif not os.path.dirname(filepath):
        filepath = output_path(filepath)
    lines = format_results_summary(results)
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    return filepath


def format_results_summary(results):
    """Format results dictionary into display-ready lines."""
    m_h = results.get("m_h", float("nan"))
    m_h_err = results.get("m_h_uncertainty", 0.0)
    theory = 125.19
    exp_val = 125.25

    theory_agree = abs(m_h - theory) / theory * 100.0 if not np.isnan(m_h) else float("nan")
    exp_agree = abs(m_h - exp_val) / exp_val * 100.0 if not np.isnan(m_h) else float("nan")

    if not np.isnan(m_h) and 125.0 <= m_h <= 125.4:
        status = "CONFIRMED"
        status_char = "+"
    elif not np.isnan(m_h) and 124.0 <= m_h <= 126.5:
        status = "MARGINAL"
        status_char = "~"
    else:
        status = "FAILED"
        status_char = "X"

    diag = results.get("diagnostics", {})
    stats = results.get("statistics", {})
    params = results.get("parameters", {})

    sep = "=" * 55
    lines = [
        sep,
        "  HIGGS MASS EXTRACTION RESULTS",
        sep,
        f"  Extracted mass:         {m_h:.2f} +/- {m_h_err:.2f} GeV",
        f"  Theoretical prediction:  125.19 GeV",
        f"  Experimental value:      125.25 +/- 0.17 GeV",
        "",
        f"  Agreement with theory:   {theory_agree:.3f}%",
        f"  Agreement with exp.:     {exp_agree:.3f}%",
        f"  Status:                  [{status_char} {status}]",
        "",
        "  DIAGNOSTIC VALUES:",
        f"  Off-diagonal mixing:     {diag.get('mixing_ratio', float('nan')):.4f}  (expected: ~0.333)",
        f"  S2 enhancement factor:   {diag.get('sphere_ratio', float('nan')):.3f}   (expected: 1.333)",
        f"  Phase sensitivity:       {diag.get('phase_sensitivity', float('nan')):.3f} GeV  (< 0.1)",
        f"  Convergence metric:      {diag.get('convergence_metric', float('nan')):.2e}  (target: < 1e-6)",
        "",
        "  INTEGRATION STATISTICS:",
        f"  Total RG time:           {stats.get('N_max', float('nan')):.2f} e-folds",
        f"  Integration steps:       {stats.get('n_steps', 0):,}",
        f"  Spiral windings:         {stats.get('windings', float('nan')):.2f}  (expected: ~15)",
        f"  Computation time:        {stats.get('cpu_time', float('nan')):.1f} seconds",
        f"  Average step size:       {stats.get('avg_step', float('nan')):.2e} e-folds",
        "",
        "  PARAMETER VALUES USED:",
        f"  theta_r = {params.get('theta_r', float('nan')):.5f}    theta_i = {params.get('theta_i', float('nan')):.5f}",
        f"  rtol    = {params.get('rtol', float('nan')):.2e}    atol    = {params.get('atol', float('nan')):.2e}",
        f"  N_max   = {params.get('N_max', float('nan')):.5f}",
        f"  G*      = {params.get('G_star', float('nan')):.5f}    Lambda* = {params.get('Lambda_star', float('nan')):.5f}",
        sep,
    ]
    return lines
