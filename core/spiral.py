"""
Spiral trajectory computation in the Reuter NGFP coupling space.

The logarithmic spiral arises from complex critical exponents theta = theta_r +/- i*theta_i.
The trajectory in (G~, Lambda~) space spirals toward the fixed point as t increases.
"""

import numpy as np
from config.parameters import G_STAR, LAMBDA_STAR, THETA_R, THETA_I, A_G, A_LAMBDA, DELTA_G, DELTA_LAMBDA


def spiral_trajectory_analytic(t_arr, G_star=G_STAR, Lambda_star=LAMBDA_STAR,
                                 theta_r=THETA_R, theta_i=THETA_I,
                                 A_G=A_G, A_Lambda=A_LAMBDA,
                                 delta_G=DELTA_G, delta_Lambda=DELTA_LAMBDA):
    """
    Analytic spiral trajectory solution.

    G~(t)      = G*  + A_G      * exp(-theta_r * t) * cos(theta_i * t + delta_G)
    Lambda~(t) = L*  + A_Lambda * exp(-theta_r * t) * sin(theta_i * t + delta_Lambda)

    Parameters
    ----------
    t_arr : array-like
        RG time values

    Returns
    -------
    G_arr : ndarray
        G~ along trajectory
    Lambda_arr : ndarray
        Lambda~ along trajectory
    """
    t = np.asarray(t_arr)
    decay = np.exp(-theta_r * t)
    phase = theta_i * t
    G_arr = G_star + A_G * decay * np.cos(phase + delta_G)
    Lambda_arr = Lambda_star + A_Lambda * decay * np.sin(phase + delta_Lambda)
    return G_arr, Lambda_arr


def distance_to_fixed_point(G_arr, Lambda_arr, G_star=G_STAR, Lambda_star=LAMBDA_STAR):
    """Euclidean distance in (G~, Lambda~) space to fixed point."""
    return np.sqrt((G_arr - G_star) ** 2 + (Lambda_arr - Lambda_star) ** 2)


def spiral_phase(t_arr, theta_i=THETA_I):
    """Phase of spiral: phi(t) = theta_i * t mod 2*pi."""
    return (theta_i * np.asarray(t_arr)) % (2.0 * np.pi)


def phase_wrapping_times(N_max, theta_i=THETA_I):
    """Return list of times t_n = 2*pi*n/theta_i within [0, N_max]."""
    T = 2.0 * np.pi / theta_i
    n_max = int(N_max / T)
    return [T * n for n in range(1, n_max + 1)]


def count_spiral_windings(t_max, theta_i=THETA_I):
    """Number of complete spiral windings from t=0 to t=t_max."""
    return t_max * theta_i / (2.0 * np.pi)


def compute_initial_conditions(G_star=G_STAR, Lambda_star=LAMBDA_STAR,
                                A_G=A_G, A_Lambda=A_LAMBDA,
                                delta_G=DELTA_G, delta_Lambda=DELTA_LAMBDA,
                                lambda_h_0=0.0, m2_h_0=0.0):
    """
    Compute initial conditions at t=0 (Planck scale) for the full ODE system.

    Returns
    -------
    y0 : ndarray of shape (4,)
        [G~(0), Lambda~(0), lambda_h(0), m2_h(0)]
    """
    G0 = G_star + A_G * np.cos(delta_G)
    Lambda0 = Lambda_star + A_Lambda * np.sin(delta_Lambda)
    return np.array([G0, Lambda0, lambda_h_0, m2_h_0])


def spiral_velocity(t, G, Lambda, G_star, Lambda_star, theta_r, theta_i):
    """
    Velocity vector (dG/dt, dLambda/dt) at point (G, Lambda).
    Useful for phase portrait visualization.
    """
    dG = -theta_r * (G - G_star) + theta_i * (Lambda - Lambda_star)
    dLam = -theta_i * (G - G_star) - theta_r * (Lambda - Lambda_star)
    return dG, dLam


def compute_trajectory_polar(t_arr, G_arr, Lambda_arr,
                              G_star=G_STAR, Lambda_star=LAMBDA_STAR,
                              theta_r=THETA_R):
    """
    Decompose the ODE trajectory into polar (amplitude, phase) coordinates
    relative to the UV fixed point (G_star, Lambda_star).

    For an ideal Reuter spiral:
      amplitude(t) = exp(-theta_r * t) * r0   →  decays to zero
      phase(t)     = -theta_i * t + phi0       →  advances monotonically
      amplitude_norm(t) = amplitude(t) * exp(theta_r * t)  ≈  const

    Parameters
    ----------
    t_arr : array-like  (n,)
    G_arr : array-like  (n,)   G~ from ODE trajectory
    Lambda_arr : array-like  (n,)   Lambda~ from ODE trajectory
    G_star, Lambda_star : float   fixed-point coordinates
    theta_r : float   real part of critical exponent

    Returns
    -------
    amplitude : ndarray (n,)
        |δu(t)| = sqrt((G-G*)^2 + (Λ-Λ*)^2)
    phase : ndarray (n,)
        Unwrapped phase = arctan2(Λ-Λ*, G-G*), monotonically advancing
    amplitude_norm : ndarray (n,)
        amplitude * exp(theta_r * t)  — should be ≈ constant for ideal spiral
    """
    t   = np.asarray(t_arr, dtype=float)
    dG  = np.asarray(G_arr, dtype=float)  - G_star
    dL  = np.asarray(Lambda_arr, dtype=float) - Lambda_star
    amplitude      = np.sqrt(dG ** 2 + dL ** 2)
    phase          = np.unwrap(np.arctan2(dL, dG))
    amplitude_norm = amplitude * np.exp(theta_r * t)
    return amplitude, phase, amplitude_norm


def phase_aligned_time(N_max, theta_i=THETA_I):
    """
    Return the largest RG time t_aligned <= N_max that corresponds to a
    complete spiral winding, i.e. t_aligned = n * 2*pi / theta_i.

    Using the phase-aligned endpoint instead of the raw endpoint t=N_max
    removes the artificial dependence of the extracted pole mass on the
    arbitrary phase at termination.

    Parameters
    ----------
    N_max : float   total RG time
    theta_i : float  imaginary part of critical exponent

    Returns
    -------
    t_aligned : float
        Last complete-winding time (<= N_max)
    n_windings : int
        Number of complete windings contained in [0, N_max]
    """
    T = 2.0 * np.pi / theta_i
    n = int(N_max / T)           # floor(N_max / T)
    return n * T, n
