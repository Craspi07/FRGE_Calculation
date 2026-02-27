"""
S^2 integration verification.

Verifies the geometric 4/3 enhancement factor by Monte Carlo integration
over the S^2 sphere vs great circle S^1.
"""

import numpy as np
from config.parameters import EXPECTED_SPHERE_RATIO


def verify_sphere_enhancement(n_samples=100000, seed=42):
    """
    Monte Carlo verification of the S^2 enhancement factor.

    Computes:
        sphere_ratio = Integral_{S^2} dOmega |nabla omega|^2
                     / Integral_{S^1} dtheta |d omega/d theta|^2

    Expected: 4/3

    The field fluctuation model used is omega(theta, phi) = cos(theta),
    corresponding to the lowest-order Goldstone mode.

    Parameters
    ----------
    n_samples : int
        Number of Monte Carlo samples
    seed : int
        Random seed for reproducibility

    Returns
    -------
    result : dict
        ratio, expected, statistical_error, acceptance
    """
    rng = np.random.default_rng(seed)

    # -----------------------------------------------------------------------
    # Full sphere S^2 integration
    # -----------------------------------------------------------------------
    # Uniform sampling on S^2 (rejection method for exact distribution)
    cos_theta = rng.uniform(-1.0, 1.0, n_samples)
    theta_s2 = np.arccos(cos_theta)
    phi_s2 = rng.uniform(0.0, 2.0 * np.pi, n_samples)

    # Goldstone field: lowest eigenmode omega = cos(theta)
    # Gradient on S^2: |nabla omega|^2 = (d omega/d theta)^2 + (d omega/d phi)^2/sin^2(theta)
    domega_dtheta = -np.sin(theta_s2)   # d(cos theta)/d theta = -sin theta
    domega_dphi = np.zeros(n_samples)   # No phi dependence for this mode

    sin_t = np.sin(theta_s2)
    sin_t_safe = np.maximum(sin_t, 1e-10)

    grad2_s2 = domega_dtheta ** 2 + domega_dphi ** 2 / sin_t_safe ** 2

    # Uniform on S^2 with cos(theta) sampling already gives uniform distribution
    # Area element: dOmega = sin(theta) d theta d phi -> already accounted for
    # by the cos_theta sampling. We just average grad2_s2 directly.
    val_s2 = np.mean(grad2_s2)
    err_s2 = np.std(grad2_s2) / np.sqrt(n_samples)

    # -----------------------------------------------------------------------
    # Great circle S^1 integration (phi = 0, theta in [0, pi])
    # -----------------------------------------------------------------------
    theta_s1 = rng.uniform(0.0, np.pi, n_samples)
    domega_dtheta_s1 = -np.sin(theta_s1)
    val_s1 = np.mean(domega_dtheta_s1 ** 2)
    err_s1 = np.std(domega_dtheta_s1 ** 2) / np.sqrt(n_samples)

    # -----------------------------------------------------------------------
    # Enhancement ratio
    # -----------------------------------------------------------------------
    # Correct normalization: full S^2 has (4pi) solid angle, S^1 has pi arc length
    # Normalize by respective geometric factors for fair comparison
    # S^2 integral: (1/4pi) * integral_S2 grad2 dOmega = val_s2 (already normalized)
    # S^1 integral: (1/pi) * integral_S1 grad2 d theta = val_s1 (already normalized)
    ratio = val_s2 / max(val_s1, 1e-30)

    return {
        "ratio": float(ratio),
        "expected": EXPECTED_SPHERE_RATIO,
        "s2_value": float(val_s2),
        "s1_value": float(val_s1),
        "samples": n_samples,
        "statistical_error": float(np.sqrt(err_s2 ** 2 + err_s1 ** 2)),
        "within_tolerance": 1.28 <= ratio <= 1.38,
        "deviation_from_expected": float(abs(ratio - EXPECTED_SPHERE_RATIO)),
    }


def analytical_sphere_enhancement():
    """
    Analytical computation of the S^2 enhancement factor.

    For the fundamental Goldstone modes on S^2 = SU(2)/U(1):
    - S^2 has 2 Goldstone degrees of freedom
    - Radial direction [0, A_0] has 1 degree of freedom
    - Total: 3 DOF out of 4 are angular (SU(2) doublet)

    The ratio of kinetic terms:
        <(nabla h)^2 + h^2 (nabla omega)^2> / <(nabla h)^2>
      = 1 + h^2/3  (for isotropic Goldstone fluctuations on S^2)
      -> 1 + 1/3 = 4/3  at h = v (VEV normalization)

    Returns 4/3.
    """
    return 4.0 / 3.0


def s2_dof_counting():
    """
    Degree-of-freedom counting for the composite manifold M = [0,A_0] x S^2.

    Returns
    -------
    summary : dict
        DOF breakdown
    """
    return {
        "total_dof": 4,          # SU(2) doublet phi_i, i=1,2,3,4
        "radial_dof": 1,         # h = |phi| (Higgs)
        "goldstone_dof": 3,      # omega_i on S^3/U(1) ~ S^2 x S^1 (3 Goldstones)
        "physical_higgs": 1,     # After EWSB
        "absorbed_goldstones": 3, # Absorbed by W+, W-, Z
        "enhancement": 4.0 / 3.0,
        "explanation": "4 DOF total -> mass enhanced by (4/3) from composite structure",
    }
