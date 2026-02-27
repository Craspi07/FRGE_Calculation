"""
Core FRGE implementation with spiral background.

Implements the Wetterich equation:
    partial_t Gamma_k = (1/2) Tr[(Gamma_k^(2) + R_k)^(-1) * partial_t R_k]

with the Litim optimized regulator R_k(p^2) = (k^2 - p^2) * theta(k^2 - p^2).
"""

import numpy as np
from .beta_functions import rhs_full_system
from .spiral import compute_initial_conditions
from config.parameters import (
    G_STAR, LAMBDA_STAR, THETA_R, THETA_I, A_G, A_LAMBDA, DELTA_G, DELTA_LAMBDA
)


def wetterich_rhs(t, y, params):
    """
    Full RHS of the Wetterich flow equation.

    State vector y = [G~, Lambda~, lambda_h, m2_h]

    The flow combines:
    1. Gravitational sector: (G~, Lambda~) spiral evolution
    2. Matter sector: Higgs (lambda_h, m2_h) running

    Parameters
    ----------
    t : float
        RG time t = ln(M_Pl / k)
    y : array of shape (4,)
        State vector
    params : dict
        Physical parameters

    Returns
    -------
    dydt : array of shape (4,)
        Time derivatives
    """
    return rhs_full_system(t, y, params)


def compute_trace_gravity(G_tilde, Lambda_tilde, k2):
    """
    Evaluate the gravitational contribution to the Wetterich trace.

    Tr[...] for TT graviton + scalar mode with Litim regulator.

    Using d=4, spherical background with Litim result:
        (1/2) Tr_{TT}[(Gamma^(2)_TT + R_k)^{-1} partial_t R_k]
        = (5 * k^4) / (16*pi^2 * (1 - 2*Lambda~)^2)

    Returns
    -------
    trace_G : float
        Graviton trace contribution
    trace_Lambda : float
        Cosmological constant trace contribution
    """
    denom = max(1.0 - 2.0 * Lambda_tilde, 0.01) ** 2

    # TT graviton (5 polarizations in d=4: traceless symmetric tensor)
    trace_TT = (5.0 * k2 ** 2) / (16.0 * np.pi ** 2 * denom) * G_tilde

    # Scalar/conformal mode
    trace_sc = -(1.0 * k2 ** 2) / (32.0 * np.pi ** 2 * denom) * G_tilde

    trace_G = trace_TT + trace_sc
    trace_Lambda = trace_TT * Lambda_tilde

    return trace_G, trace_Lambda


def compute_trace_scalar(lambda_h, m2_h, k2):
    """
    Evaluate the scalar (Higgs) contribution to the Wetterich trace.

    With Litim regulator:
        (1/2) Tr_{scalar}[...] = k^4 / (16*pi^2 * (1 + m^2/k^2))

    Returns
    -------
    trace_scalar : float
        Scalar contribution to flow
    """
    w = m2_h   # Dimensionless mass parameter (m^2/k^2 already)
    denom = max(1.0 + w, 0.01) ** 2
    return (1.0 / (32.0 * np.pi ** 2)) * k2 ** 2 / denom


def litim_threshold_scalar(w, n=0):
    """
    Litim threshold function l^n_0(w) = 1 / (1+w)^(n+1).
    Arises from the Litim regulator momentum integral.
    """
    return 1.0 / (1.0 + w) ** (n + 1)


def litim_threshold_graviton(w, n=0):
    """
    Litim threshold function for the graviton sector.
    """
    return 1.0 / (1.0 + w) ** (n + 1)


def get_default_params():
    """Return default parameter dictionary for the ODE system."""
    return {
        "theta_r": THETA_R,
        "theta_i": THETA_I,
        "G_star": G_STAR,
        "Lambda_star": LAMBDA_STAR,
        "A_G": A_G,
        "A_Lambda": A_LAMBDA,
        "delta_G": DELTA_G,
        "delta_Lambda": DELTA_LAMBDA,
    }


def get_initial_conditions(params=None):
    """
    Compute initial conditions at t=0 (Planck scale).

    Returns
    -------
    y0 : ndarray of shape (4,)
        [G~(0), Lambda~(0), lambda_h(0), m2_h(0)]
    """
    if params is None:
        params = get_default_params()
    return compute_initial_conditions(
        G_star=params["G_star"],
        Lambda_star=params["Lambda_star"],
        A_G=params["A_G"],
        A_Lambda=params["A_Lambda"],
        delta_G=params["delta_G"],
        delta_Lambda=params["delta_Lambda"],
        lambda_h_0=0.0,   # Shaposhnikov-Wetterich: lambda=0 at Planck scale
        m2_h_0=0.0,
    )
