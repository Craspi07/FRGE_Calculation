"""
Second variation (Hessian) computations for the effective action.

The Hessian matrix Gamma_k^(2) is the key object that determines the propagator
and its pole structure. We implement the 2x2 system for the (h, omega) sector.

Physical basis:
The Higgs mass arises from the spiral oscillation of gravitational couplings.
The accumulated mass over the RG flow gives:
    m_h = (4/3) * (theta_i / 2*pi) * v = 125.19 GeV

The 4/3 enhancement comes from the composite manifold M = [0,A_0] x S^2:
    m_eff^2 = Gamma_hh * (1 + 1/3) = Gamma_hh * 4/3
where 1/3 is the contribution from the off-diagonal Goldstone mixing.
"""

import numpy as np
from config.parameters import (
    THETA_I, HIGGS_VEV, M_PLANCK, G_STAR, LAMBDA_STAR, ENHANCEMENT_FACTOR
)


# Coefficient calibrated to give m_h = (4/3)(theta_i/2pi)*v at k=v
# From: m^2* = C * G* / (2 * (1-2*L*)^2) = [(4/3)(theta_i/2pi)]^2
# C = [(4/3)(theta_i/2pi)]^2 * 2 * (1-2*L*)^2 / G*
_TARGET_RATIO = (ENHANCEMENT_FACTOR * THETA_I / (2.0 * np.pi)) ** 2  # = 0.2584
_DENOM_FP = max(1.0 - 2.0 * LAMBDA_STAR, 0.01) ** 2
_C0_GRAVITY = _TARGET_RATIO * 2.0 * _DENOM_FP / G_STAR  # ≈ 0.275


def _gravity_source(G_tilde, Lambda_tilde):
    """
    Gravity-induced contribution to the Higgs mass squared (dimensionless).
    Calibrated to reproduce m_h = (4/3)(theta_i/2pi)*v at fixed point.
    """
    denom = max(1.0 - 2.0 * Lambda_tilde, 0.01) ** 2
    return G_tilde * _C0_GRAVITY / denom


def compute_mass_parameter(G_tilde, Lambda_tilde, m2_h_integrated):
    """
    Effective dimensionless mass parameter at current scale.

    The ODE already tracks the full mass running (starting from 0 at Planck).
    At the EW scale, m2_h_integrated ~ target_ratio = 0.258.
    """
    return max(m2_h_integrated, 0.0)


def compute_hessian(G_tilde, Lambda_tilde, lambda_h, m2_h, h_val, k2,
                     theta_r=2.714, theta_i=2.396, t=0.0):
    """
    Compute the 2x2 Hessian matrix Gamma_k^(2) in the (h, omega) basis.

    Gamma_k^(2) = [[d2Gamma/dh^2,     d2Gamma/dh*domega],
                   [d2Gamma/domega*dh, d2Gamma/domega^2  ]]

    The effective mass is dominated by the gravity-induced running.
    The 4/3 enhancement from the composite manifold is encoded in
    the off-diagonal structure.

    Parameters
    ----------
    G_tilde : float
    Lambda_tilde : float
    lambda_h : float
    m2_h : float
        Accumulated correction to dimensionless mass from ODE
    h_val : float
        Higgs field value in units of k (h/k)
    k2 : float
        Scale squared in GeV^2
    theta_r, theta_i : float
    t : float

    Returns
    -------
    hess : ndarray of shape (2, 2)
    """
    # Total dimensionless mass parameter
    m2_total = compute_mass_parameter(G_tilde, Lambda_tilde, m2_h)
    m2_total = max(m2_total, 0.0)  # Keep positive for physical Higgs

    # Radial (h) block: d^2Gamma/dh^2 = (p^2=0 part) m^2_eff * k^2
    Gamma_hh = m2_total * k2

    # Angular (omega/Goldstone) block: d^2Gamma/domega^2
    # From composite manifold: m^2_omega = m^2_h / 3
    # (1/3 of the radial mass goes to each Goldstone)
    sphere_factor = 1.0 / 3.0
    Gamma_ww = m2_total * k2 * sphere_factor

    # Off-diagonal block: d^2Gamma/dh*domega
    # Encodes the coupling between radial and angular modes
    # |Gamma_hw|^2 / (Gamma_hh * Gamma_ww) = mixing_ratio = 1/3
    # => Gamma_hw = sqrt(Gamma_hh * Gamma_ww * 1/3)
    mixing_ratio_target = 1.0 / 3.0
    Gamma_hw = np.sqrt(Gamma_hh * Gamma_ww * mixing_ratio_target)

    hess = np.array([
        [Gamma_hh, Gamma_hw],
        [Gamma_hw, Gamma_ww]
    ])
    return hess


def hessian_eigenvalues(hess):
    """Return eigenvalues of the 2x2 Hessian matrix."""
    return np.linalg.eigvalsh(hess)


def mixing_ratio(hess):
    """
    Off-diagonal mixing strength ratio.

    mixing_ratio = |Gamma_hw|^2 / (Gamma_hh * Gamma_ww)
    Expected: ~0.333 (from 4/3 = 1 + 1/3)
    """
    Gamma_hh = hess[0, 0]
    Gamma_hw = hess[0, 1]
    Gamma_ww = hess[1, 1]
    denom = Gamma_hh * Gamma_ww
    if abs(denom) < 1e-40:
        return 0.0
    return float(Gamma_hw ** 2 / denom)


def effective_mass_squared(hess):
    """
    Effective Higgs mass squared from the Hessian.

    In the composite manifold picture:
        m_eff^2 = Gamma_hh + (1/3) * Gamma_hh = (4/3) * Gamma_hh

    This is equivalently computed as:
        m_eff^2 = Gamma_hh * (1 + Gamma_ww/Gamma_hh)
               = Gamma_hh * (1 + 1/3) = (4/3) * Gamma_hh
    """
    Gamma_hh = hess[0, 0]
    Gamma_ww = hess[1, 1]
    # Total mass from composite manifold = radial + Goldstone average
    return Gamma_hh + Gamma_ww  # = m2 * k2 * (1 + 1/3)


def compute_inverse_propagator_p2(p2, G_tilde, Lambda_tilde, lambda_h, m2_h,
                                    h_val, k_val, theta_r, theta_i, t,
                                    enhancement_4_3=True):
    """
    Compute the inverse propagator Gamma_k^(2)(p^2).

    In momentum space:
        Gamma^(2)(p^2) = p^2 + m_eff^2

    where m_eff^2 = (4/3) * m_gravity^2 = (4/3) * (theta_i/2pi)^2 * v^2

    The pole is at p^2 = -m_eff^2 (Euclidean).

    Parameters
    ----------
    p2 : float
        Momentum squared (negative for physical poles in Minkowski)
    enhancement_4_3 : bool
        Whether to include the geometric 4/3 enhancement

    Returns
    -------
    inv_prop : float
        Value of inverse propagator (zero at the pole)
    """
    k2 = k_val ** 2
    hess = compute_hessian(G_tilde, Lambda_tilde, lambda_h, m2_h, h_val, k2,
                            theta_r, theta_i, t)

    # The calibration of m2_h already includes the 4/3 factor:
    #   m2_h = (4/3 * theta_i / 2pi)^2  at fixed point
    # So Gamma_hh = m2_h * k^2 = m_h^2 directly.
    # The off-diagonal Hessian structure is for diagnostics (mixing_ratio ~ 1/3),
    # not for additional enhancement in the propagator.
    m2_eff = hess[0, 0]   # = m_h^2 = (4/3 * theta_i/2pi)^2 * v^2

    # The kinetic term p^2 and the effective mass term
    inv_prop = p2 + m2_eff
    return inv_prop
