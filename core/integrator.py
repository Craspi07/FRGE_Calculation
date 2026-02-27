"""
Adaptive RG flow integration using DOP853 (8th-order Runge-Kutta).

Integrates the combined gravitational + scalar system from the Planck scale
(t=0) to the electroweak scale (t=N_max ~ 39.34 e-folds).
"""

import time
import numpy as np
from scipy.integrate import solve_ivp

from .wetterich import wetterich_rhs, get_default_params, get_initial_conditions
from .spiral import phase_wrapping_times, count_spiral_windings
from config.parameters import (
    DEFAULT_RTOL, DEFAULT_ATOL, DEFAULT_N_MAX, DEFAULT_MAX_STEP,
    RESONANCE_N_MAX, T_RG, M_PLANCK, HIGGS_VEV
)
from utils.validation import validate_parameters, validate_integration_result


def run_integration(
    theta_r=None,
    theta_i=None,
    N_max=None,
    rtol=DEFAULT_RTOL,
    atol=DEFAULT_ATOL,
    termination="fixed",
    params=None,
    progress_callback=None,
):
    """
    Run the full FRGE integration from Planck scale to EW scale.

    Parameters
    ----------
    theta_r : float, optional
        Real part of critical exponent (default: 2.714)
    theta_i : float, optional
        Imaginary part of critical exponent (default: 2.396)
    N_max : float, optional
        Total RG time (default: ln(M_Pl/v) ~ 39.34)
    rtol : float
        Relative tolerance for ODE solver
    atol : float
        Absolute tolerance for ODE solver
    termination : str
        'fixed' | 'resonance' | 'variational'
    params : dict, optional
        Override parameter dictionary
    progress_callback : callable, optional
        Called with (t, y) during integration for progress monitoring

    Returns
    -------
    result : dict
        Contains 't', 'y', 'statistics', 'success', 'message'
    """
    # Build parameter dict
    if params is None:
        params = get_default_params()
    if theta_r is not None:
        params["theta_r"] = theta_r
    if theta_i is not None:
        params["theta_i"] = theta_i

    _theta_r = params["theta_r"]
    _theta_i = params["theta_i"]

    # Determine N_max
    if termination == "fixed":
        _N_max = N_max if N_max is not None else DEFAULT_N_MAX
    elif termination == "resonance":
        _N_max = RESONANCE_N_MAX if theta_i is None else 30.0 * np.pi / _theta_i
    elif termination == "variational":
        _N_max = N_max if N_max is not None else DEFAULT_N_MAX
    else:
        _N_max = N_max if N_max is not None else DEFAULT_N_MAX

    # Validate parameters
    validate_parameters(_theta_i, _theta_r, _N_max, rtol, atol)

    # Initial conditions
    y0 = get_initial_conditions(params)

    # Step size control
    max_step = T_RG / 100.0 if T_RG else DEFAULT_MAX_STEP

    # Phase wrapping event functions (to add dense output near wrapping)
    wrap_times = phase_wrapping_times(_N_max, _theta_i)

    # Timing
    t_start = time.time()

    # Progress tracking via dense_output
    t_eval = None  # Use dense output

    # ODE definition
    def ode_rhs(t, y):
        return wetterich_rhs(t, y, params)

    # Run integration with DOP853
    try:
        sol = solve_ivp(
            ode_rhs,
            t_span=(0.0, _N_max),
            y0=y0,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            dense_output=True,
        )
    except Exception as e:
        return {
            "success": False,
            "message": str(e),
            "t": np.array([0.0]),
            "y": np.array([y0]).T,
            "statistics": {},
        }

    t_end = time.time()
    cpu_time = t_end - t_start

    # Validate result
    try:
        validate_integration_result(sol)
    except RuntimeError as e:
        return {
            "success": False,
            "message": str(e),
            "t": sol.t,
            "y": sol.y,
            "statistics": {},
        }

    # Compute statistics
    step_sizes = np.diff(sol.t)
    windings = count_spiral_windings(_N_max, _theta_i)

    statistics = {
        "N_max": _N_max,
        "n_steps": len(sol.t),
        "windings": windings,
        "cpu_time": cpu_time,
        "avg_step": np.mean(step_sizes),
        "min_step": np.min(step_sizes),
        "max_step": np.max(step_sizes),
        "step_ratio": np.max(step_sizes) / max(np.min(step_sizes), 1e-30),
        "n_evals": sol.nfev,
        "wrap_times": wrap_times,
    }

    return {
        "success": True,
        "message": sol.message,
        "t": sol.t,
        "y": sol.y,
        "sol": sol,    # Keep dense solution for interpolation
        "statistics": statistics,
        "params": params,
    }


def run_integration_with_mass_evolution(theta_r=None, theta_i=None, N_max=None,
                                          rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL,
                                          termination="fixed", params=None):
    """
    Run integration and compute mass evolution m_h(t) at each step.

    Returns integration result augmented with mass evolution array.
    """
    from .pole_finder import extract_pole_at_scale

    result = run_integration(
        theta_r=theta_r, theta_i=theta_i, N_max=N_max,
        rtol=rtol, atol=atol, termination=termination, params=params
    )

    if not result["success"]:
        return result

    # Compute mass at subsample of time points
    t_arr = result["t"]
    y_arr = result["y"]

    # Sample every 50th point for efficiency
    stride = max(1, len(t_arr) // 200)
    t_sample = t_arr[::stride]
    y_sample = y_arr[:, ::stride]

    mass_evolution = []
    for i in range(len(t_sample)):
        t_i = t_sample[i]
        y_i = y_sample[:, i]
        k_val = M_PLANCK * np.exp(-t_i)
        try:
            m_i = extract_pole_at_scale(y_i, k_val, result.get("params", {}))
        except Exception:
            m_i = float("nan")
        mass_evolution.append(m_i)

    result["t_mass"] = t_sample
    result["mass_evolution"] = np.array(mass_evolution)
    return result
