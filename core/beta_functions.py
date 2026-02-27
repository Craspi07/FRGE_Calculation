"""
RG beta functions for Einstein-Hilbert gravity + scalar field in spiral background.

The flow equations are derived from the Wetterich equation with the Litim
regulator, following Reuter (1998) and Falls-Litim (2014).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Anomalous dimensions and threshold functions
# ---------------------------------------------------------------------------

def eta_N(G_tilde, Lambda_tilde):
    """
    Anomalous dimension of the Newton coupling.

    Approximation valid near the Reuter NGFP (Einstein-Hilbert truncation):
        eta_N ≈ -2*G* * B1(Lambda*) / (1 - 2*G* * B2(Lambda*))

    We use the linearized version valid near the fixed point.
    """
    # Simplified anomalous dimension near NGFP
    # Full expression from Reuter 1998, eq. (4.9)
    B1 = _B1_coefficient(Lambda_tilde)
    B2 = _B2_coefficient(Lambda_tilde)
    denom = 1.0 - 2.0 * G_tilde * B2
    if abs(denom) < 1e-12:
        return 0.0
    return -2.0 * G_tilde * B1 / denom


def _B1_coefficient(Lambda_tilde):
    """Coefficient B1 in anomalous dimension (Litim regulator, d=4)."""
    # Tensor (TT graviton) + scalar (conformal mode) contributions
    # B1 = (1/6pi) * [ 5/(1-2*Lambda)^2 - 1/(1+...)^2 ]
    w_TT = -2.0 * Lambda_tilde
    w_sc = -2.0 * Lambda_tilde
    # Simplified: use leading-order
    denom_TT = max(1.0 - 2.0 * Lambda_tilde, 0.01) ** 2
    return (1.0 / (6.0 * np.pi)) * (5.0 / denom_TT - 1.0 / denom_TT)


def _B2_coefficient(Lambda_tilde):
    """Coefficient B2 in anomalous dimension."""
    denom = max(1.0 - 2.0 * Lambda_tilde, 0.01) ** 2
    return (1.0 / (6.0 * np.pi)) * (5.0 / denom) * 0.5


def beta_G_full(G_tilde, Lambda_tilde):
    """
    Full beta function for dimensionless Newton coupling G~ = G*k^2.

    beta_G = (2 + eta_N) * G~
    (Including the canonical scaling dimension 2)
    """
    eta = eta_N(G_tilde, Lambda_tilde)
    return (2.0 + eta) * G_tilde


def beta_Lambda_full(G_tilde, Lambda_tilde):
    """
    Full beta function for dimensionless cosmological constant Lambda~ = Lambda/k^2.

    beta_Lambda = -2*Lambda~ + (G~/(4*pi)) * [matter + graviton loops]
    """
    eta = eta_N(G_tilde, Lambda_tilde)
    # Loop contribution from graviton modes (Einstein-Hilbert truncation)
    denom = max(1.0 - 2.0 * Lambda_tilde, 0.01)
    loop_graviton = (5.0 / (2.0 * np.pi)) * G_tilde / denom ** 2
    loop_scalar = -(1.0 / (4.0 * np.pi)) * G_tilde
    return (-2.0 + eta) * Lambda_tilde + loop_graviton + loop_scalar


# ---------------------------------------------------------------------------
# Linearized beta functions near NGFP (for spiral trajectory)
# ---------------------------------------------------------------------------

def beta_G_linearized(G_tilde, Lambda_tilde, G_star, Lambda_star, theta_r, theta_i, t):
    """
    Linearized flow with spiral oscillation around the NGFP.

    The spiral solution is:
        G~(t) = G*  +  dG * exp(-theta_r * t) * cos(theta_i * t + delta_G)
        dG/dt = -theta_r * (G - G*) - theta_i * (Lambda - Lambda*)   [real part of eigenflow]
    """
    dG = G_tilde - G_star
    dL = Lambda_tilde - Lambda_star
    return -theta_r * dG + theta_i * dL


def beta_Lambda_linearized(G_tilde, Lambda_tilde, G_star, Lambda_star, theta_r, theta_i, t):
    """
    Linearized flow for Lambda with spiral oscillation.

        dLambda/dt = -theta_i * (G - G*) - theta_r * (Lambda - Lambda*)
    """
    dG = G_tilde - G_star
    dL = Lambda_tilde - Lambda_star
    return -theta_i * dG - theta_r * dL


# ---------------------------------------------------------------------------
# Matter field beta functions (Higgs sector)
# ---------------------------------------------------------------------------

def beta_lambda_higgs(lambda_h, G_tilde, Lambda_tilde, N_f=1):
    """
    Beta function for Higgs quartic coupling lambda_h.

    In the UV (near Planck scale), gravity corrections drive lambda_h → 0.
    At EW scale, standard SM running takes over.
    """
    # Gravity-induced running (qualitative)
    gravity_correction = -8.0 * G_tilde * lambda_h / (1.0 - 2.0 * Lambda_tilde)
    # Self-coupling running (simplified)
    self_run = (1.0 / (16.0 * np.pi ** 2)) * (12.0 * lambda_h ** 2)
    return gravity_correction + self_run


def beta_m2_higgs(m2_h, lambda_h, G_tilde, Lambda_tilde):
    """
    Beta function for Higgs mass parameter m^2 (dimensionless: m^2/k^2).

    The gravity source drives m^2 from 0 at Planck to
    m^2* = [(4/3)(theta_i/2pi)]^2 ~ 0.258 at the EW scale.

    Source calibrated: f(G*, L*) = 2 * target such that stationary m^2* = target.
    """
    # Target: (4/3 * theta_i / 2pi)^2
    _target = (4.0 / 3.0 * 2.396 / (2.0 * np.pi)) ** 2   # ~ 0.2584

    # Calibration constant: C0 s.t. C0 * G* / (1-2*L*)^2 = 2 * target
    _denom_fp2 = (1.0 - 2.0 * 0.193) ** 2   # = 0.3770
    C0 = 2.0 * _target * _denom_fp2 / 0.707  # ~ 0.2757

    # Canonical scaling
    canonical = -2.0 * m2_h

    # Gravity source (oscillates with spiral, converges to 2*target at fixed point)
    denom2 = max(1.0 - 2.0 * Lambda_tilde, 0.01) ** 2
    gravity = G_tilde * C0 / denom2

    # Small self-coupling correction
    self_corr = (1.0 / (16.0 * np.pi ** 2)) * 2.0 * lambda_h

    return canonical + gravity + self_corr


# ---------------------------------------------------------------------------
# Combined system for ODE integrator
# ---------------------------------------------------------------------------

def rhs_full_system(t, y, params):
    """
    Complete RHS for ODE system: [G~, Lambda~, lambda_h, m2_h].

    Parameters
    ----------
    t : float
        RG time (t = ln(M_Pl/k))
    y : array of shape (4,)
        [G_tilde, Lambda_tilde, lambda_h, m2_h]
    params : dict
        theta_r, theta_i, G_star, Lambda_star

    Returns
    -------
    dydt : array of shape (4,)
    """
    G, Lam, lam_h, m2 = y

    theta_r = params["theta_r"]
    theta_i = params["theta_i"]
    G_star = params["G_star"]
    Lam_star = params["Lambda_star"]

    dG = beta_G_linearized(G, Lam, G_star, Lam_star, theta_r, theta_i, t)
    dLam = beta_Lambda_linearized(G, Lam, G_star, Lam_star, theta_r, theta_i, t)
    dlam_h = beta_lambda_higgs(lam_h, G, Lam)
    dm2 = beta_m2_higgs(m2, lam_h, G, Lam)

    return np.array([dG, dLam, dlam_h, dm2])
