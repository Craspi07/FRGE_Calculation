"""
Off-diagonal Hessian term diagnostics.

Analyzes the mixing between radial Higgs (h) and angular Goldstone (omega)
modes throughout the RG flow.
"""

import numpy as np
from core.hessian import compute_hessian, mixing_ratio as compute_mixing_ratio
from config.parameters import (
    G_STAR, LAMBDA_STAR, THETA_R, THETA_I, M_PLANCK, HIGGS_VEV,
    EXPECTED_MIXING_RATIO
)


def analyze_mixing_term(integration_result, subsample=100):
    """
    Compute d^2Gamma/dh*domega at each RG step and analyze its evolution.

    Expected behavior:
    - Early times (t~0): mixing ~ 0 (Planck scale, no composite structure)
    - Late times (t~N_max): mixing ~ 0.333 (EW scale, full 4/3 enhancement)

    Parameters
    ----------
    integration_result : dict
        Result from run_integration()
    subsample : int
        Evaluate at every Nth step for efficiency

    Returns
    -------
    result : dict
        mixing_evolution, final_mixing, deviation from expected
    """
    if not integration_result.get("success", False):
        return {"error": "Integration result not available"}

    t_arr = integration_result["t"]
    y_arr = integration_result["y"]
    params = integration_result.get("params", {})

    theta_r = params.get("theta_r", THETA_R)
    theta_i = params.get("theta_i", THETA_I)

    # Subsample
    idx = np.arange(0, len(t_arr), max(1, len(t_arr) // subsample))
    t_sub = t_arr[idx]
    y_sub = y_arr[:, idx]

    mixing_evolution = []
    hh_evolution = []
    ww_evolution = []
    hw_evolution = []

    for i in range(len(t_sub)):
        t_i = t_sub[i]
        y_i = y_sub[:, i]

        G_tilde = y_i[0]
        Lambda_tilde = y_i[1]
        lambda_h = max(y_i[2], 0.0)
        m2_h = y_i[3]

        k_val = M_PLANCK * np.exp(-t_i)
        k2 = k_val ** 2
        h_val = HIGGS_VEV / k_val

        hess = compute_hessian(G_tilde, Lambda_tilde, lambda_h, m2_h,
                                h_val, k2, theta_r, theta_i, t_i)

        mix = compute_mixing_ratio(hess)
        mixing_evolution.append(mix)
        hh_evolution.append(hess[0, 0])
        ww_evolution.append(hess[1, 1])
        hw_evolution.append(hess[0, 1])

    mixing_evolution = np.array(mixing_evolution)
    final_mixing = mixing_evolution[-1] if len(mixing_evolution) > 0 else float("nan")

    return {
        "t_arr": t_sub,
        "mixing_evolution": mixing_evolution,
        "hh_evolution": np.array(hh_evolution),
        "ww_evolution": np.array(ww_evolution),
        "hw_evolution": np.array(hw_evolution),
        "final_mixing": float(final_mixing),
        "expected": EXPECTED_MIXING_RATIO,
        "deviation": float(abs(final_mixing - EXPECTED_MIXING_RATIO)),
        "within_tolerance": 0.28 <= final_mixing <= 0.38,
    }


def compute_enhancement_factor_from_mixing(mixing_ratio_val):
    """
    Compute the mass enhancement factor from the off-diagonal mixing ratio.

    Enhancement = 1 + mixing_ratio ~ 4/3 when mixing_ratio ~ 1/3
    """
    return 1.0 + mixing_ratio_val
