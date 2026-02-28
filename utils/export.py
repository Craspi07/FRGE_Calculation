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


def _pf(condition):
    """Return '[PASS]' or '[FAIL]' string for display."""
    return "[PASS]" if condition else "[FAIL]"


def format_results_summary(results):
    """
    Format results dictionary into display-ready lines with pass/fail indicators.

    Success criteria (Part 7 of specification):
      m_h ∈ [125.1, 125.3] GeV                → CONFIRMED
      Off-diagonal mixing ratio ≈ 1/3 ± 0.05  → PASS
      S² enhancement ≈ 4/3 ± 0.05             → PASS
      Phase sensitivity < 0.1 GeV             → PASS
      Convergence Δm_h < 0.01 GeV             → PASS
    """
    from config.parameters import (
        HIGGS_MASS_TARGET_MIN, HIGGS_MASS_TARGET_MAX,
        HIGGS_MASS_MARGINAL_MIN, HIGGS_MASS_MARGINAL_MAX,
        MIXING_RATIO_MIN, MIXING_RATIO_MAX, MIXING_RATIO_TARGET,
        SPHERE_RATIO_MIN, SPHERE_RATIO_MAX, SPHERE_RATIO_TARGET,
        PHASE_SENSITIVITY_THRESHOLD, CONVERGENCE_THRESHOLD,
    )

    m_h     = results.get("m_h", float("nan"))
    m_h_err = results.get("m_h_uncertainty", 0.0)
    theory  = 125.19
    exp_val = 125.25

    theory_agree = abs(m_h - theory) / theory * 100.0 if not np.isnan(m_h) else float("nan")
    exp_agree    = abs(m_h - exp_val) / exp_val * 100.0 if not np.isnan(m_h) else float("nan")

    # Status using spec-defined window [125.1, 125.3]
    if not np.isnan(m_h) and HIGGS_MASS_TARGET_MIN <= m_h <= HIGGS_MASS_TARGET_MAX:
        status, status_char = "CONFIRMED", "+"
    elif not np.isnan(m_h) and HIGGS_MASS_MARGINAL_MIN <= m_h <= HIGGS_MASS_MARGINAL_MAX:
        status, status_char = "MARGINAL", "~"
    else:
        status, status_char = "FAILED", "X"

    diag   = results.get("diagnostics", {})
    stats  = results.get("statistics", {})
    params = results.get("parameters", {})

    mix   = diag.get("mixing_ratio",      float("nan"))
    sph   = diag.get("sphere_ratio",      float("nan"))
    phase = diag.get("phase_sensitivity", float("nan"))
    conv  = diag.get("convergence_metric", float("nan"))

    mass_ok  = not np.isnan(m_h)    and HIGGS_MASS_TARGET_MIN <= m_h    <= HIGGS_MASS_TARGET_MAX
    mix_ok   = not np.isnan(mix)    and MIXING_RATIO_MIN       <= mix    <= MIXING_RATIO_MAX
    sph_ok   = not np.isnan(sph)    and SPHERE_RATIO_MIN        <= sph   <= SPHERE_RATIO_MAX
    phase_ok = not np.isnan(phase)  and phase < PHASE_SENSITIVITY_THRESHOLD
    conv_ok  = not np.isnan(conv)   and conv  < CONVERGENCE_THRESHOLD

    rtol = params.get("rtol", float("nan"))
    atol = params.get("atol", float("nan"))
    rtol_s = f"{rtol:.2e}" if not np.isnan(rtol) else "---"
    atol_s = f"{atol:.2e}" if not np.isnan(atol) else "---"

    sep = "=" * 64
    lines = [
        sep,
        "  HIGGS MASS EXTRACTION RESULTS",
        sep,
        f"  Extracted mass:            {m_h:.4f} +/- {m_h_err:.4f} GeV   {_pf(mass_ok)}",
        f"  Theoretical prediction:    {theory:.2f} GeV  (4/3 × θ_i/2π × v)",
        f"  Experimental value:        {exp_val:.2f} ± 0.17 GeV  (PDG 2023)",
        f"  Confirmed window:          [{HIGGS_MASS_TARGET_MIN:.1f}, {HIGGS_MASS_TARGET_MAX:.1f}] GeV",
        "",
        f"  Agreement with theory:     {theory_agree:.4f}%",
        f"  Agreement with exp.:       {exp_agree:.4f}%",
        f"  Status:                    [{status_char} {status}]",
        "",
        "  DIAGNOSTIC VALUES                   TARGET          RESULT",
        "  " + "-" * 60,
        f"  Off-diagonal mixing:       {mix:7.4f}   ~{MIXING_RATIO_TARGET:.3f} ± 0.05   {_pf(mix_ok)}",
        f"  S² enhancement factor:     {sph:7.4f}   ~{SPHERE_RATIO_TARGET:.3f} ± 0.05   {_pf(sph_ok)}",
        f"  Phase sensitivity:         {phase:7.4f} GeV   < {PHASE_SENSITIVITY_THRESHOLD:.1f} GeV      {_pf(phase_ok)}",
        f"  Convergence Δm_h:          {conv:.2e}   < {CONVERGENCE_THRESHOLD:.0e}         {_pf(conv_ok)}",
        "",
        "  INTEGRATION STATISTICS:",
        f"  Total RG time:             {stats.get('N_max', float('nan')):.4f} e-folds",
        f"  Integration steps:         {stats.get('n_steps', 0):,}",
        f"  Spiral windings:           {stats.get('windings', float('nan')):.2f}   (expected: ~15)",
        f"  Computation time:          {stats.get('cpu_time', float('nan')):.2f} s",
        f"  Avg / Min / Max step:      {stats.get('avg_step', float('nan')):.2e} / {stats.get('min_step', float('nan')):.2e} / {stats.get('max_step', float('nan')):.2e} e-folds",
        "",
        "  PARAMETER VALUES USED:",
        f"  θ_r   = {params.get('theta_r', float('nan')):.5f}    θ_i  = {params.get('theta_i', float('nan')):.5f}",
        f"  rtol  = {rtol_s}    atol = {atol_s}",
        f"  N_max = {params.get('N_max', float('nan')):.5f} e-folds",
        f"  G*    = {params.get('G_star', float('nan')):.5f}    Λ*   = {params.get('Lambda_star', float('nan')):.5f}",
        f"  A_G   = {params.get('A_G', float('nan')):.5f}    A_Λ  = {params.get('A_Lambda', float('nan')):.5f}",
        sep,
    ]
    return lines
