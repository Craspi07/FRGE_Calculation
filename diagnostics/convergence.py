"""
Tolerance testing and convergence validation.

Verifies that the extracted Higgs mass is stable with respect to
tightening integration tolerances.
"""

import time
import numpy as np
from core.integrator import run_integration
from core.pole_finder import extract_pole
from config.parameters import DEFAULT_N_MAX, THETA_R, THETA_I


TOLERANCE_LEVELS = [
    (1e-5, 1e-7),    # Coarse
    (1e-6, 1e-8),    # Standard
    (1e-7, 1e-9),    # Fine
    (1e-8, 1e-10),   # Very fine
    (1e-9, 1e-11),   # Ultra fine
]


def run_convergence_test(theta_r=THETA_R, theta_i=THETA_I, N_max=DEFAULT_N_MAX,
                          tolerance_levels=None, convergence_threshold=0.005,
                          progress_callback=None):
    """
    Multi-level convergence verification.

    Runs integration at increasing tolerance levels and checks that
    the extracted mass converges.

    Parameters
    ----------
    theta_r, theta_i : float
        Critical exponents
    N_max : float
        Total RG time
    tolerance_levels : list of (rtol, atol) tuples, optional
    convergence_threshold : float
        Mass convergence criterion in GeV (default: 5 MeV)
    progress_callback : callable, optional
        Called with (level_index, total_levels, current_mass) for progress

    Returns
    -------
    result : dict
        Convergence status, final mass, uncertainty, timing
    """
    if tolerance_levels is None:
        tolerance_levels = TOLERANCE_LEVELS

    masses = []
    times_list = []
    uncertainties = []
    details = []

    for idx, (rtol, atol) in enumerate(tolerance_levels):
        if progress_callback:
            progress_callback(idx, len(tolerance_levels), masses[-1] if masses else float("nan"))

        t0 = time.time()
        try:
            result = run_integration(
                theta_r=theta_r, theta_i=theta_i, N_max=N_max,
                rtol=rtol, atol=atol
            )
            pole_result = extract_pole(result)
            m_h = pole_result.get("m_h", float("nan"))
            unc = pole_result.get("m_h_uncertainty", float("nan"))
        except Exception as e:
            m_h = float("nan")
            unc = float("nan")
            result = {"success": False, "message": str(e), "statistics": {}}

        elapsed = time.time() - t0
        masses.append(m_h)
        times_list.append(elapsed)
        uncertainties.append(unc)

        detail = {
            "rtol": rtol,
            "atol": atol,
            "m_h": m_h,
            "uncertainty": unc,
            "cpu_time": elapsed,
            "n_steps": result.get("statistics", {}).get("n_steps", 0),
            "success": result.get("success", False),
        }
        details.append(detail)

        # Check convergence once we have 3 levels
        if len(masses) >= 3:
            valid = [m for m in masses[-3:] if not np.isnan(m)]
            if len(valid) >= 2:
                recent_std = np.std(valid)
                if recent_std < convergence_threshold:
                    final_mass = masses[-1]
                    return {
                        "status": "CONVERGED",
                        "final_mass": final_mass,
                        "uncertainty": recent_std,
                        "recommended_tolerance": (rtol, atol),
                        "mass_trend": masses,
                        "time_cost": times_list,
                        "details": details,
                        "n_levels_run": idx + 1,
                    }

    # Did not converge
    valid_masses = [m for m in masses if not np.isnan(m)]
    final_std = np.std(valid_masses) if len(valid_masses) >= 2 else float("nan")

    return {
        "status": "NEEDS_TIGHTER_TOLERANCE",
        "final_mass": masses[-1] if masses else float("nan"),
        "uncertainty": final_std,
        "recommended_tolerance": tolerance_levels[-1],
        "mass_trend": masses,
        "time_cost": times_list,
        "details": details,
        "n_levels_run": len(tolerance_levels),
    }


def compute_convergence_metric(masses):
    """
    Compute convergence metric from sequence of masses.
    Returns standard deviation of last 3 values (or all if fewer).
    """
    valid = [m for m in masses if not np.isnan(m)]
    if len(valid) < 2:
        return float("nan")
    window = valid[-3:]
    return float(np.std(window))


def format_convergence_report(result):
    """Format convergence test results as a human-readable string."""
    lines = ["CONVERGENCE TEST REPORT", "=" * 40]
    lines.append(f"Status: {result['status']}")
    lines.append(f"Final mass: {result.get('final_mass', float('nan')):.3f} GeV")
    lines.append(f"Uncertainty: {result.get('uncertainty', float('nan')):.4f} GeV")
    lines.append("")
    lines.append(f"{'Level':<6} {'rtol':<12} {'atol':<12} {'m_h (GeV)':<12} {'Time (s)':<10}")
    lines.append("-" * 60)
    for d in result.get("details", []):
        lines.append(
            f"  {'C' if d['success'] else 'F'}    {d['rtol']:.2e}    {d['atol']:.2e}"
            f"    {d['m_h']:.4f}     {d['cpu_time']:.2f}"
        )
    return "\n".join(lines)
