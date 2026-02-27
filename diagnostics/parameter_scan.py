"""
Multi-dimensional parameter space exploration.

Performs a 2D scan over (theta_i, N_max) space and generates
a contour map of the extracted Higgs mass.
"""

import numpy as np
from joblib import Parallel, delayed

from core.integrator import run_integration
from core.pole_finder import extract_pole
from config.parameters import (
    THETA_R, THETA_I, DEFAULT_N_MAX, DEFAULT_RTOL, DEFAULT_ATOL,
    SCAN_THETA_I_MIN, SCAN_THETA_I_MAX, SCAN_THETA_I_POINTS,
    SCAN_N_MAX_MIN, SCAN_N_MAX_MAX, SCAN_N_MAX_POINTS,
    MC_THETA_I_SIGMA, MC_N_SAMPLES
)


def _run_single(theta_i, N_max, theta_r, rtol, atol):
    """Run a single integration and return extracted mass."""
    try:
        result = run_integration(
            theta_r=theta_r, theta_i=theta_i, N_max=N_max,
            rtol=rtol, atol=atol
        )
        pole_result = extract_pole(result)
        return float(pole_result.get("m_h", float("nan")))
    except Exception:
        return float("nan")


def run_parameter_scan(
    theta_i_range=None,
    N_max_range=None,
    theta_r=THETA_R,
    rtol=1e-6,
    atol=1e-8,
    n_jobs=-1,
    progress_callback=None,
):
    """
    2D parameter scan over (theta_i, N_max) grid.

    Parameters
    ----------
    theta_i_range : tuple (min, max, n_points), optional
    N_max_range : tuple (min, max, n_points), optional
    theta_r : float
        Fixed theta_r value
    rtol, atol : float
        Integration tolerances (looser for speed)
    n_jobs : int
        Number of parallel jobs (-1 = all CPUs)
    progress_callback : callable, optional

    Returns
    -------
    result : dict
        theta_i_arr, N_max_arr, mass_grid, sensitivity metrics
    """
    if theta_i_range is None:
        theta_i_range = (SCAN_THETA_I_MIN, SCAN_THETA_I_MAX, SCAN_THETA_I_POINTS)
    if N_max_range is None:
        N_max_range = (SCAN_N_MAX_MIN, SCAN_N_MAX_MAX, SCAN_N_MAX_POINTS)

    theta_i_arr = np.linspace(*theta_i_range)
    N_max_arr = np.linspace(*N_max_range)

    # Create parameter pairs
    pairs = [(ti, nm) for ti in theta_i_arr for nm in N_max_arr]
    total = len(pairs)

    # Run in parallel
    masses_flat = Parallel(n_jobs=n_jobs)(
        delayed(_run_single)(ti, nm, theta_r, rtol, atol)
        for ti, nm in pairs
    )

    # Reshape to grid
    mass_grid = np.array(masses_flat).reshape(len(theta_i_arr), len(N_max_arr))

    # Compute sensitivities
    dmh_dtheta_i = np.gradient(mass_grid, theta_i_arr, axis=0)
    dmh_dN_max = np.gradient(mass_grid, N_max_arr, axis=1)

    # Find sweet spot (closest to 125.25 GeV)
    diff = np.abs(mass_grid - 125.25)
    idx = np.unravel_index(np.nanargmin(diff), mass_grid.shape)
    sweet_theta_i = theta_i_arr[idx[0]]
    sweet_N_max = N_max_arr[idx[1]]

    return {
        "theta_i_arr": theta_i_arr,
        "N_max_arr": N_max_arr,
        "mass_grid": mass_grid,
        "dmh_dtheta_i": dmh_dtheta_i,
        "dmh_dN_max": dmh_dN_max,
        "sweet_spot": {
            "theta_i": sweet_theta_i,
            "N_max": sweet_N_max,
            "m_h": float(mass_grid[idx]),
        },
        "n_total": total,
    }


def run_monte_carlo_uncertainty(
    theta_r=THETA_R,
    theta_i_mean=THETA_I,
    theta_i_sigma=MC_THETA_I_SIGMA,
    N_max_mean=DEFAULT_N_MAX,
    n_samples=MC_N_SAMPLES,
    rtol=1e-6,
    atol=1e-8,
    n_jobs=-1,
    seed=42,
    progress_callback=None,
):
    """
    Monte Carlo uncertainty propagation.

    Samples theta_i ~ Normal(theta_i_mean, theta_i_sigma) and
    runs integration for each sample to estimate final mass uncertainty.

    Returns
    -------
    result : dict
        m_h_mean, m_h_std, m_h_samples, parameter_samples
    """
    rng = np.random.default_rng(seed)
    theta_i_samples = rng.normal(theta_i_mean, theta_i_sigma, n_samples)
    # Clip to physical range
    theta_i_samples = np.clip(theta_i_samples, 1.5, 4.0)

    masses = Parallel(n_jobs=n_jobs)(
        delayed(_run_single)(ti, N_max_mean, theta_r, rtol, atol)
        for ti in theta_i_samples
    )

    masses = np.array(masses)
    valid = masses[~np.isnan(masses)]

    return {
        "m_h_mean": float(np.mean(valid)) if len(valid) > 0 else float("nan"),
        "m_h_std": float(np.std(valid)) if len(valid) > 1 else float("nan"),
        "m_h_median": float(np.median(valid)) if len(valid) > 0 else float("nan"),
        "m_h_samples": masses,
        "theta_i_samples": theta_i_samples,
        "n_valid": int(len(valid)),
        "n_total": n_samples,
    }
