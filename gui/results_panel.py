"""
Formatted results display for the GUI.

Generates the comprehensive results summary panel content.
"""

import numpy as np
from config.parameters import (
    HIGGS_MASS_THEORY, HIGGS_MASS_EXP, HIGGS_MASS_ERROR,
    HIGGS_MASS_TARGET_MIN, HIGGS_MASS_TARGET_MAX,
    HIGGS_MASS_MARGINAL_MIN, HIGGS_MASS_MARGINAL_MAX,
)
from utils.export import format_results_summary


def build_results_dict(integration_result, pole_result, mixing_result=None,
                       sphere_result=None, phase_result=None, convergence_result=None):
    """
    Build a unified results dictionary from all computation outputs.

    Parameters
    ----------
    integration_result : dict
        From run_integration()  – must contain "params", "statistics",
        and optionally "run_params" (dict with rtol, atol).
    pole_result : dict
        From extract_pole()
    mixing_result : dict, optional
        From analyze_mixing_term()
    sphere_result : dict, optional
        From verify_sphere_enhancement()
    phase_result : dict, optional
        From check_phase_sensitivity()
    convergence_result : dict, optional
        From run_convergence_test()

    Returns
    -------
    results : dict
        Unified results dictionary ready for format_results_summary().
    """
    m_h     = pole_result.get("m_h",             float("nan")) if pole_result else float("nan")
    m_h_unc = pole_result.get("m_h_uncertainty",  float("nan")) if pole_result else float("nan")

    params = (integration_result or {}).get("params",     {})
    stats  = (integration_result or {}).get("statistics", {})
    # rtol / atol stored by run_integration in "run_params"
    run_p  = (integration_result or {}).get("run_params", {})

    # ── Diagnostics ───────────────────────────────────────────────────────────
    diag = {}

    if mixing_result:
        diag["mixing_ratio"] = mixing_result.get("final_mixing", float("nan"))
    else:
        pole_diag = (pole_result or {}).get("diagnostics", {})
        diag["mixing_ratio"] = pole_diag.get("mixing_ratio", float("nan"))

    diag["sphere_ratio"]      = sphere_result.get("ratio",       float("nan")) if sphere_result else float("nan")
    diag["phase_sensitivity"] = phase_result.get("sensitivity",  float("nan")) if phase_result  else float("nan")
    diag["convergence_metric"] = convergence_result.get("uncertainty", float("nan")) if convergence_result else float("nan")

    return {
        "m_h":             m_h,
        "m_h_uncertainty": m_h_unc,
        "diagnostics": diag,
        "statistics": {
            "N_max":    stats.get("N_max",    float("nan")),
            "n_steps":  stats.get("n_steps",  0),
            "windings": stats.get("windings", float("nan")),
            "cpu_time": stats.get("cpu_time", float("nan")),
            "avg_step": stats.get("avg_step", float("nan")),
            "min_step": stats.get("min_step", float("nan")),
            "max_step": stats.get("max_step", float("nan")),
        },
        "parameters": {
            "theta_r":     params.get("theta_r",     float("nan")),
            "theta_i":     params.get("theta_i",     float("nan")),
            "rtol":        run_p.get("rtol",          float("nan")),
            "atol":        run_p.get("atol",          float("nan")),
            "N_max":       stats.get("N_max",         float("nan")),
            "G_star":      params.get("G_star",       0.707),
            "Lambda_star": params.get("Lambda_star",  0.193),
            "A_G":         params.get("A_G",          float("nan")),
            "A_Lambda":    params.get("A_Lambda",     float("nan")),
        },
    }


def get_status(m_h):
    """Return (status_string, color) for a given extracted mass."""
    if np.isnan(m_h):
        return "FAILED", "red"
    if HIGGS_MASS_TARGET_MIN <= m_h <= HIGGS_MASS_TARGET_MAX:
        return "CONFIRMED", "green"
    if HIGGS_MASS_MARGINAL_MIN <= m_h <= HIGGS_MASS_MARGINAL_MAX:
        return "MARGINAL", "orange"
    return "FAILED", "red"


def format_full_summary(results):
    """Return formatted summary lines for display."""
    return format_results_summary(results)
