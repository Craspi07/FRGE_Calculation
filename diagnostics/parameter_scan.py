"""
Multi-dimensional parameter space exploration.

Public API
----------
run_parameter_scan()      – original 2-D grid (theta_i × N_max)
run_full_grid_scan()      – 3-D grid (theta_r × theta_i × N_max)
run_sensitivity_scan()    – independent 1-D sweep for every parameter
run_monte_carlo_uncertainty() – MC uncertainty propagation
"""

import numpy as np
from joblib import Parallel, delayed

from core.integrator import run_integration
from core.pole_finder import extract_pole
from config.parameters import (
    THETA_R, THETA_I, DEFAULT_N_MAX, DEFAULT_RTOL, DEFAULT_ATOL,
    A_G, A_LAMBDA, DELTA_LAMBDA,
    SCAN_THETA_I_MIN, SCAN_THETA_I_MAX, SCAN_THETA_I_POINTS,
    SCAN_N_MAX_MIN,   SCAN_N_MAX_MAX,   SCAN_N_MAX_POINTS,
    SCAN_THETA_R_MIN, SCAN_THETA_R_MAX, SCAN_THETA_R_POINTS,
    SCAN_A_G_MIN,     SCAN_A_G_MAX,     SCAN_A_G_POINTS,
    SCAN_A_LAMBDA_MIN, SCAN_A_LAMBDA_MAX, SCAN_A_LAMBDA_POINTS,
    SCAN_DELTA_LAMBDA_MIN, SCAN_DELTA_LAMBDA_MAX, SCAN_DELTA_LAMBDA_POINTS,
    MC_THETA_I_SIGMA, MC_N_SAMPLES,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _run_single(theta_i, N_max, theta_r, rtol, atol):
    """Run a single integration and return extracted mass (float)."""
    try:
        result = run_integration(
            theta_r=theta_r, theta_i=theta_i, N_max=N_max,
            rtol=rtol, atol=atol
        )
        pole_result = extract_pole(result)
        return float(pole_result.get("m_h", float("nan")))
    except Exception:
        return float("nan")


def _run_single_full(theta_r, theta_i, N_max, A_G_val, A_Lambda_val,
                     delta_Lambda_val, rtol, atol):
    """
    Single integration with all physics parameters specified.
    Used by run_sensitivity_scan and run_full_grid_scan.
    """
    try:
        from core.wetterich import get_default_params
        params = get_default_params()
        params["theta_r"]      = theta_r
        params["theta_i"]      = theta_i
        params["A_G"]          = A_G_val
        params["A_Lambda"]     = A_Lambda_val
        params["delta_Lambda"] = delta_Lambda_val

        result = run_integration(
            theta_r=theta_r, theta_i=theta_i, N_max=N_max,
            rtol=rtol, atol=atol, params=params
        )
        pole_result = extract_pole(result)
        return float(pole_result.get("m_h", float("nan")))
    except Exception:
        return float("nan")


# ── 2-D scan (original) ───────────────────────────────────────────────────────

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
    2-D parameter scan over (theta_i, N_max) grid.

    Parameters
    ----------
    theta_i_range : tuple (min, max, n_points), optional
    N_max_range   : tuple (min, max, n_points), optional
    theta_r       : float  – fixed theta_r value
    rtol, atol    : float  – integration tolerances (looser for speed)
    n_jobs        : int    – parallel workers (-1 = all CPUs)

    Returns
    -------
    dict with keys: theta_i_arr, N_max_arr, mass_grid,
                    dmh_dtheta_i, dmh_dN_max, sweet_spot, n_total
    """
    if theta_i_range is None:
        theta_i_range = (SCAN_THETA_I_MIN, SCAN_THETA_I_MAX, SCAN_THETA_I_POINTS)
    if N_max_range is None:
        N_max_range = (SCAN_N_MAX_MIN, SCAN_N_MAX_MAX, SCAN_N_MAX_POINTS)

    theta_i_arr = np.linspace(*theta_i_range)
    N_max_arr   = np.linspace(*N_max_range)

    pairs = [(ti, nm) for ti in theta_i_arr for nm in N_max_arr]
    total = len(pairs)

    masses_flat = Parallel(n_jobs=n_jobs)(
        delayed(_run_single)(ti, nm, theta_r, rtol, atol)
        for ti, nm in pairs
    )

    mass_grid = np.array(masses_flat).reshape(len(theta_i_arr), len(N_max_arr))

    dmh_dtheta_i = np.gradient(mass_grid, theta_i_arr, axis=0)
    dmh_dN_max   = np.gradient(mass_grid, N_max_arr,   axis=1)

    diff = np.abs(mass_grid - 125.25)
    idx  = np.unravel_index(np.nanargmin(diff), mass_grid.shape)

    return {
        "theta_i_arr":   theta_i_arr,
        "N_max_arr":     N_max_arr,
        "mass_grid":     mass_grid,
        "dmh_dtheta_i":  dmh_dtheta_i,
        "dmh_dN_max":    dmh_dN_max,
        "sweet_spot": {
            "theta_i": float(theta_i_arr[idx[0]]),
            "N_max":   float(N_max_arr[idx[1]]),
            "m_h":     float(mass_grid[idx]),
        },
        "n_total": total,
        "scan_type": "2d_grid",
    }


# ── 3-D grid scan ─────────────────────────────────────────────────────────────

def run_full_grid_scan(
    theta_r_range=None,
    theta_i_range=None,
    N_max_range=None,
    A_G_val=A_G,
    A_Lambda_val=A_LAMBDA,
    delta_Lambda_val=DELTA_LAMBDA,
    rtol=1e-6,
    atol=1e-8,
    n_jobs=-1,
):
    """
    3-D parameter scan over (theta_r × theta_i × N_max) with all other
    parameters fixed at their nominal values.

    Returns
    -------
    dict with keys:
        theta_r_arr, theta_i_arr, N_max_arr,
        mass_cube  – shape (n_r, n_i, n_N),
        sweet_spot – best (theta_r, theta_i, N_max) and m_h,
        projections – three 2-D mean projections for plotting,
        n_total, scan_type
    """
    if theta_r_range is None:
        theta_r_range = (SCAN_THETA_R_MIN, SCAN_THETA_R_MAX, SCAN_THETA_R_POINTS)
    if theta_i_range is None:
        theta_i_range = (SCAN_THETA_I_MIN, SCAN_THETA_I_MAX, SCAN_THETA_I_POINTS)
    if N_max_range is None:
        N_max_range = (SCAN_N_MAX_MIN, SCAN_N_MAX_MAX, SCAN_N_MAX_POINTS)

    theta_r_arr = np.linspace(*theta_r_range)
    theta_i_arr = np.linspace(*theta_i_range)
    N_max_arr   = np.linspace(*N_max_range)

    triples = [
        (tr, ti, nm)
        for tr in theta_r_arr
        for ti in theta_i_arr
        for nm in N_max_arr
    ]
    total = len(triples)

    masses_flat = Parallel(n_jobs=n_jobs)(
        delayed(_run_single_full)(
            tr, ti, nm, A_G_val, A_Lambda_val, delta_Lambda_val, rtol, atol
        )
        for tr, ti, nm in triples
    )

    mass_cube = np.array(masses_flat).reshape(
        len(theta_r_arr), len(theta_i_arr), len(N_max_arr)
    )

    # Best point
    diff = np.abs(mass_cube - 125.25)
    idx  = np.unravel_index(np.nanargmin(diff), mass_cube.shape)

    # 2-D projections (nanmean over the third axis)
    proj_ri  = np.nanmean(mass_cube, axis=2)   # theta_r vs theta_i
    proj_rN  = np.nanmean(mass_cube, axis=1)   # theta_r vs N_max
    proj_iN  = np.nanmean(mass_cube, axis=0)   # theta_i vs N_max

    return {
        "theta_r_arr": theta_r_arr,
        "theta_i_arr": theta_i_arr,
        "N_max_arr":   N_max_arr,
        "mass_cube":   mass_cube,
        "sweet_spot": {
            "theta_r": float(theta_r_arr[idx[0]]),
            "theta_i": float(theta_i_arr[idx[1]]),
            "N_max":   float(N_max_arr[idx[2]]),
            "m_h":     float(mass_cube[idx]),
        },
        "projections": {
            "theta_r_vs_theta_i": proj_ri,
            "theta_r_vs_N_max":   proj_rN,
            "theta_i_vs_N_max":   proj_iN,
        },
        "n_total":   total,
        "scan_type": "3d_grid",
    }


# ── 1-D sensitivity scan (every parameter independently) ─────────────────────

def run_sensitivity_scan(
    rtol=1e-6,
    atol=1e-8,
    n_jobs=-1,
):
    """
    Independent 1-D sweep of every physics parameter.

    For each parameter the remaining parameters are held at their nominal
    (default) values.  Returns a dict of {param_name: {values, masses}}
    suitable for direct plotting.

    Parameters swept
    ----------------
    theta_r     : [SCAN_THETA_R_MIN  … SCAN_THETA_R_MAX]
    theta_i     : [SCAN_THETA_I_MIN  … SCAN_THETA_I_MAX]
    N_max       : [SCAN_N_MAX_MIN    … SCAN_N_MAX_MAX]
    A_G         : [SCAN_A_G_MIN      … SCAN_A_G_MAX]
    A_Lambda    : [SCAN_A_LAMBDA_MIN … SCAN_A_LAMBDA_MAX]
    delta_Lambda: [SCAN_DELTA_LAMBDA_MIN … SCAN_DELTA_LAMBDA_MAX]

    Returns
    -------
    dict  – one entry per parameter, each containing:
        "values"   : 1-D array of parameter values swept
        "masses"   : 1-D array of extracted m_h values
        "label"    : human-readable axis label
        "nominal"  : nominal (default) value
    """
    # Default (nominal) values
    nom = dict(
        theta_r      = THETA_R,
        theta_i      = THETA_I,
        N_max        = DEFAULT_N_MAX,
        A_G          = A_G,
        A_Lambda     = A_LAMBDA,
        delta_Lambda = DELTA_LAMBDA,
    )

    sweeps = {
        "theta_r":      (np.linspace(SCAN_THETA_R_MIN,       SCAN_THETA_R_MAX,       SCAN_THETA_R_POINTS),
                         r"$\theta_r$"),
        "theta_i":      (np.linspace(SCAN_THETA_I_MIN,       SCAN_THETA_I_MAX,       SCAN_THETA_I_POINTS),
                         r"$\theta_i$"),
        "N_max":        (np.linspace(SCAN_N_MAX_MIN,          SCAN_N_MAX_MAX,          SCAN_N_MAX_POINTS),
                         r"$N_{\max}$ (e-folds)"),
        "A_G":          (np.linspace(SCAN_A_G_MIN,            SCAN_A_G_MAX,            SCAN_A_G_POINTS),
                         r"$A_G$"),
        "A_Lambda":     (np.linspace(SCAN_A_LAMBDA_MIN,       SCAN_A_LAMBDA_MAX,       SCAN_A_LAMBDA_POINTS),
                         r"$A_\Lambda$"),
        "delta_Lambda": (np.linspace(SCAN_DELTA_LAMBDA_MIN,   SCAN_DELTA_LAMBDA_MAX,   SCAN_DELTA_LAMBDA_POINTS),
                         r"$\delta_\Lambda$ (rad)"),
    }

    results = {}
    for param_name, (values, label) in sweeps.items():

        def _job(v, _p=param_name):
            kw = dict(nom)        # copy nominals
            kw[_p] = v            # override swept parameter
            return _run_single_full(
                kw["theta_r"], kw["theta_i"], kw["N_max"],
                kw["A_G"], kw["A_Lambda"], kw["delta_Lambda"],
                rtol, atol,
            )

        masses = Parallel(n_jobs=n_jobs)(delayed(_job)(v) for v in values)

        results[param_name] = {
            "values":  values,
            "masses":  np.array(masses, dtype=float),
            "label":   label,
            "nominal": nom[param_name],
        }

    return {
        "sweeps":    results,
        "nominal":   nom,
        "scan_type": "sensitivity_1d",
    }


# ── Monte Carlo ───────────────────────────────────────────────────────────────

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
    dict with m_h_mean, m_h_std, m_h_samples, theta_i_samples, n_valid, n_total
    """
    rng = np.random.default_rng(seed)
    theta_i_samples = rng.normal(theta_i_mean, theta_i_sigma, n_samples)
    theta_i_samples = np.clip(theta_i_samples, 1.5, 4.0)

    masses = Parallel(n_jobs=n_jobs)(
        delayed(_run_single)(ti, N_max_mean, theta_r, rtol, atol)
        for ti in theta_i_samples
    )

    masses = np.array(masses)
    valid  = masses[~np.isnan(masses)]

    return {
        "m_h_mean":        float(np.mean(valid))   if len(valid) > 0 else float("nan"),
        "m_h_std":         float(np.std(valid))    if len(valid) > 1 else float("nan"),
        "m_h_median":      float(np.median(valid)) if len(valid) > 0 else float("nan"),
        "m_h_samples":     masses,
        "theta_i_samples": theta_i_samples,
        "n_valid":         int(len(valid)),
        "n_total":         n_samples,
    }
