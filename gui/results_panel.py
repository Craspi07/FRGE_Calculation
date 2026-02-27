"""
Formatted results display for the GUI.

Generates the comprehensive results summary panel content.
"""

import numpy as np
from config.parameters import HIGGS_MASS_THEORY, HIGGS_MASS_EXP, HIGGS_MASS_ERROR
from utils.export import format_results_summary


def build_results_dict(integration_result, pole_result, mixing_result=None,
                        sphere_result=None, phase_result=None, convergence_result=None):
    """
    Build a unified results dictionary from all computation outputs.

    Parameters
    ----------
    integration_result : dict
        From run_integration()
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
        Unified results dictionary
    """
    m_h = pole_result.get("m_h", float("nan")) if pole_result else float("nan")
    m_h_unc = pole_result.get("m_h_uncertainty", float("nan")) if pole_result else float("nan")

    params = integration_result.get("params", {}) if integration_result else {}
    stats = integration_result.get("statistics", {}) if integration_result else {}

    # Diagnostics
    diag = {}
    if mixing_result:
        diag["mixing_ratio"] = mixing_result.get("final_mixing", float("nan"))
    else:
        # From pole result diagnostics
        pole_diag = pole_result.get("diagnostics", {}) if pole_result else {}
        diag["mixing_ratio"] = pole_diag.get("mixing_ratio", float("nan"))

    if sphere_result:
        diag["sphere_ratio"] = sphere_result.get("ratio", float("nan"))
    else:
        diag["sphere_ratio"] = 4.0 / 3.0  # Expected value placeholder

    if phase_result:
        diag["phase_sensitivity"] = phase_result.get("sensitivity", float("nan"))
    else:
        diag["phase_sensitivity"] = float("nan")

    if convergence_result:
        conv_metric = convergence_result.get("uncertainty", float("nan"))
    else:
        conv_metric = float("nan")
    diag["convergence_metric"] = conv_metric

    return {
        "m_h": m_h,
        "m_h_uncertainty": m_h_unc,
        "diagnostics": diag,
        "statistics": {
            "N_max": stats.get("N_max", float("nan")),
            "n_steps": stats.get("n_steps", 0),
            "windings": stats.get("windings", float("nan")),
            "cpu_time": stats.get("cpu_time", float("nan")),
            "avg_step": stats.get("avg_step", float("nan")),
            "min_step": stats.get("min_step", float("nan")),
            "max_step": stats.get("max_step", float("nan")),
        },
        "parameters": {
            "theta_r": params.get("theta_r", float("nan")),
            "theta_i": params.get("theta_i", float("nan")),
            "rtol": float("nan"),
            "atol": float("nan"),
            "N_max": stats.get("N_max", float("nan")),
            "G_star": params.get("G_star", 0.707),
            "Lambda_star": params.get("Lambda_star", 0.193),
        },
    }


def get_status(m_h):
    """Return status string and color for given extracted mass."""
    if np.isnan(m_h):
        return "FAILED", "red"
    if 125.0 <= m_h <= 125.4 and abs(m_h - HIGGS_MASS_EXP) <= HIGGS_MASS_ERROR:
        return "CONFIRMED", "green"
    if 124.0 <= m_h <= 126.5:
        return "MARGINAL", "orange"
    return "FAILED", "red"


def format_full_summary(results):
    """Return formatted summary lines for display."""
    return format_results_summary(results)
