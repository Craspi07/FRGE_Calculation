"""
Phase sensitivity analysis.

Checks whether the extracted Higgs mass depends unphysically on the
termination phase of the RG trajectory.
"""

import numpy as np
from core.integrator import run_integration
from core.pole_finder import extract_pole
from config.parameters import THETA_R, THETA_I, DEFAULT_RTOL, DEFAULT_ATOL


def check_phase_sensitivity(theta_r=THETA_R, theta_i=THETA_I,
                              n_phases=36, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL,
                              progress_callback=None):
    """
    Vary termination phase over one complete spiral period and measure mass variation.

    Physical requirement: std(m_h) < 0.1 GeV over a complete period.
    If larger, the result has unphysical dependence on the arbitrary cutoff.

    Parameters
    ----------
    theta_r, theta_i : float
        Critical exponents
    n_phases : int
        Number of phase points (default: 36 = every 10 degrees)
    rtol, atol : float
        Integration tolerances
    progress_callback : callable, optional
        Called with (i, total, current_mass) for progress monitoring

    Returns
    -------
    result : dict
        mass_variation, sensitivity, acceptable
    """
    phases = np.linspace(0.0, 2.0 * np.pi, n_phases, endpoint=False)
    masses = []

    for i, delta_phi in enumerate(phases):
        N_term = (30.0 * np.pi + delta_phi) / theta_i

        if progress_callback:
            m_prev = masses[-1] if masses else float("nan")
            progress_callback(i, n_phases, m_prev)

        try:
            result = run_integration(
                theta_r=theta_r, theta_i=theta_i, N_max=N_term,
                rtol=rtol, atol=atol
            )
            pole_result = extract_pole(result)
            m_h = pole_result.get("m_h", float("nan"))
        except Exception:
            m_h = float("nan")

        masses.append(m_h)

    masses = np.array(masses)
    valid_mask = ~np.isnan(masses)
    valid = masses[valid_mask]

    sensitivity = float(np.std(valid)) if len(valid) > 1 else float("nan")
    max_dev = float(np.max(valid) - np.min(valid)) if len(valid) > 1 else float("nan")
    mean_mass = float(np.mean(valid)) if len(valid) > 0 else float("nan")

    return {
        "phases": phases,
        "phases_deg": np.degrees(phases),
        "masses": masses,
        "valid_masses": valid,
        "sensitivity": sensitivity,
        "max_deviation": max_dev,
        "mean_mass": mean_mass,
        "acceptable": sensitivity < 0.1 if not np.isnan(sensitivity) else False,
        "n_valid": int(np.sum(valid_mask)),
    }
