"""
Root finding for propagator poles.

Extracts the Higgs mass from the zero of the inverse propagator
using Brent's method with cubic spline interpolation.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from config.parameters import (
    POLE_SEARCH_MIN, POLE_SEARCH_MAX, POLE_SEARCH_POINTS,
    G_STAR, LAMBDA_STAR, THETA_R, THETA_I, M_PLANCK, HIGGS_VEV
)
from .hessian import compute_inverse_propagator_p2


def extract_pole(integration_result, enhancement_4_3=True):
    """
    Extract the Higgs pole mass from a completed integration result.

    Parameters
    ----------
    integration_result : dict
        Result from run_integration()
    enhancement_4_3 : bool
        Whether to include geometric 4/3 enhancement

    Returns
    -------
    result : dict
        Contains 'm_h', 'p2_pole', 'uncertainty', 'diagnostics'
    """
    if not integration_result.get("success", False):
        return {"m_h": float("nan"), "success": False, "message": "Integration failed"}

    t_arr = integration_result["t"]
    y_arr = integration_result["y"]
    params = integration_result.get("params", {})

    # Use the final state at t = N_max
    t_final = t_arr[-1]
    y_final = y_arr[:, -1]

    k_final = M_PLANCK * np.exp(-t_final)

    result = extract_pole_at_scale(
        y_final, k_final, params,
        enhancement_4_3=enhancement_4_3,
        return_diagnostics=True
    )

    result["t_final"] = t_final
    result["k_final"] = k_final
    return result


def extract_pole_at_scale(y_state, k_val, params, enhancement_4_3=True,
                           return_diagnostics=False):
    """
    Extract pole mass at a given RG scale k.

    Parameters
    ----------
    y_state : array of shape (4,)
        [G~, Lambda~, lambda_h, m2_h] at scale k
    k_val : float
        RG scale in GeV
    params : dict
        Physical parameters
    enhancement_4_3 : bool
        Whether to include 4/3 enhancement

    Returns
    -------
    result : dict or float
        Pole mass result (dict if return_diagnostics=True)
    """
    G_tilde = y_state[0]
    Lambda_tilde = y_state[1]
    lambda_h = max(y_state[2], 0.0)
    m2_h = y_state[3]
    h_val = HIGGS_VEV / k_val  # Higgs VEV in units of k

    theta_r = params.get("theta_r", THETA_R)
    theta_i = params.get("theta_i", THETA_I)
    t_val = np.log(M_PLANCK / k_val)

    # Search domain in p^2 (GeV^2)
    # For a particle of mass m_h, pole at p^2 = -m_h^2 (Minkowski)
    p2_min = POLE_SEARCH_MIN
    p2_max = POLE_SEARCH_MAX

    # Evaluate inverse propagator on search grid
    p2_arr = np.linspace(p2_min, p2_max, POLE_SEARCH_POINTS)
    inv_prop_arr = np.array([
        compute_inverse_propagator_p2(
            p2, G_tilde, Lambda_tilde, lambda_h, m2_h,
            h_val, k_val, theta_r, theta_i, t_val,
            enhancement_4_3=enhancement_4_3
        )
        for p2 in p2_arr
    ])

    # Find zero crossing
    p2_pole = _find_zero_brent_spline(p2_arr, inv_prop_arr)

    if p2_pole is None or p2_pole >= 0:
        # Try wider search range
        p2_arr_wide = np.linspace(-135.0 ** 2, -115.0 ** 2, POLE_SEARCH_POINTS * 2)
        inv_prop_wide = np.array([
            compute_inverse_propagator_p2(
                p2, G_tilde, Lambda_tilde, lambda_h, m2_h,
                h_val, k_val, theta_r, theta_i, t_val,
                enhancement_4_3=enhancement_4_3
            )
            for p2 in p2_arr_wide
        ])
        p2_pole = _find_zero_brent_spline(p2_arr_wide, inv_prop_wide)

    if p2_pole is None or p2_pole >= 0:
        if not return_diagnostics:
            return float("nan")
        return {
            "m_h": float("nan"),
            "p2_pole": float("nan"),
            "success": False,
            "message": "No pole found in search range",
            "p2_arr": p2_arr,
            "inv_prop_arr": inv_prop_arr,
        }

    m_h = np.sqrt(-p2_pole)  # Euclidean -> Minkowski

    if not return_diagnostics:
        return m_h

    # Compute uncertainty estimate
    uncertainty = _estimate_pole_uncertainty(p2_arr, inv_prop_arr, p2_pole)

    # Diagnostics
    from .hessian import compute_hessian, mixing_ratio as compute_mixing_ratio

    k2 = k_val ** 2
    hess = compute_hessian(G_tilde, Lambda_tilde, lambda_h, m2_h, h_val, k2,
                            theta_r, theta_i, t_val)
    mix_ratio = compute_mixing_ratio(hess)

    return {
        "m_h": m_h,
        "m_h_uncertainty": uncertainty,
        "p2_pole": p2_pole,
        "success": True,
        "message": "Pole found successfully",
        "p2_arr": p2_arr,
        "inv_prop_arr": inv_prop_arr,
        "diagnostics": {
            "mixing_ratio": mix_ratio,
            "hessian": hess,
            "G_tilde": G_tilde,
            "Lambda_tilde": Lambda_tilde,
            "lambda_h": lambda_h,
            "m2_h": m2_h,
        },
    }


def _find_zero_brent_spline(x_arr, y_arr):
    """Find zero crossing using cubic spline + Brent method."""
    # Find sign changes
    signs = np.sign(y_arr)
    sign_changes = np.where(np.diff(signs) != 0)[0]

    if len(sign_changes) == 0:
        return None

    try:
        cs = CubicSpline(x_arr, y_arr)
    except Exception:
        return None

    # Use first sign change (closest to expected pole)
    i = sign_changes[0]
    a, b = x_arr[i], x_arr[i + 1]

    try:
        x_zero = brentq(cs, a, b, xtol=1e-8, rtol=1e-10, maxiter=200)
        return x_zero
    except ValueError:
        return None


def _estimate_pole_uncertainty(p2_arr, inv_prop_arr, p2_pole):
    """
    Estimate uncertainty in pole location from curvature of inverse propagator.
    Returns uncertainty in m_h (GeV).
    """
    try:
        cs = CubicSpline(p2_arr, inv_prop_arr)
        deriv = cs(p2_pole, 1)  # First derivative at pole
        if abs(deriv) < 1e-20 or p2_pole >= 0:
            return 0.05
        # delta_p2 ~ delta_Gamma / |d Gamma/d p^2|
        # Assume numerical precision ~ 1e-8 * |Gamma_max|
        delta_gamma = 1e-8 * max(np.max(np.abs(inv_prop_arr)), 1.0)
        delta_p2 = delta_gamma / abs(deriv)
        # Propagate: delta_m = delta_p2 / (2 * m)
        m_h = np.sqrt(-p2_pole)
        delta_m = delta_p2 / (2.0 * m_h) if m_h > 0 else 0.05
        return min(delta_m, 1.0)  # Cap at 1 GeV
    except Exception:
        return 0.05


def scan_pole_vs_termination(base_params, theta_i=None, n_phases=36,
                               rtol=1e-8, atol=1e-10):
    """
    Scan the extracted pole mass vs termination phase over one complete period.

    Used for phase sensitivity analysis.
    """
    from .integrator import run_integration

    if theta_i is None:
        theta_i = base_params.get("theta_i", THETA_I)

    phases = np.linspace(0, 2.0 * np.pi, n_phases, endpoint=False)
    masses = []

    for delta_phi in phases:
        N_term = (30.0 * np.pi + delta_phi) / theta_i
        result = run_integration(N_max=N_term, rtol=rtol, atol=atol,
                                  params=dict(base_params))
        if result["success"]:
            pole_result = extract_pole(result)
            masses.append(pole_result.get("m_h", float("nan")))
        else:
            masses.append(float("nan"))

    masses = np.array(masses)
    valid = masses[~np.isnan(masses)]
    sensitivity = np.std(valid) if len(valid) > 1 else float("nan")

    return {
        "phases": phases,
        "masses": masses,
        "sensitivity": sensitivity,
        "max_deviation": np.max(valid) - np.min(valid) if len(valid) > 1 else float("nan"),
        "acceptable": sensitivity < 0.1 if not np.isnan(sensitivity) else False,
    }
