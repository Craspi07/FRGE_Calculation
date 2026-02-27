"""Mathematical utilities and special functions."""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq


def litim_regulator(p2, k2):
    """
    Litim optimized regulator R_k(p^2).
    R_k(p^2) = (k^2 - p^2) * theta(k^2 - p^2)
    """
    return np.maximum(k2 - p2, 0.0)


def litim_regulator_dot(p2, k2):
    """
    Scale derivative of Litim regulator: partial_t R_k = 2*k^2 * theta(k^2 - p^2)
    (with t = ln(k), so partial_t = k * partial_k = 2*k^2)
    """
    return np.where(p2 < k2, 2.0 * k2, 0.0)


def threshold_function_0(w):
    """
    Litim threshold function l^0_n(w) = 1/(1+w).
    Appears in scalar fluctuation traces.
    """
    return 1.0 / (1.0 + w)


def threshold_function_1(w):
    """
    Litim threshold function l^1_n(w) = 1/(1+w)^2.
    Appears in graviton fluctuation traces.
    """
    return 1.0 / (1.0 + w) ** 2


def sphere_solid_angle(n):
    """Surface area of unit n-sphere."""
    from scipy.special import gamma
    return 2.0 * np.pi ** ((n + 1) / 2.0) / gamma((n + 1) / 2.0)


def compute_spiral_phase(t, theta_i):
    """Compute spiral phase phi = theta_i * t mod 2*pi."""
    return (theta_i * t) % (2.0 * np.pi)


def find_zero_cubic_spline(x_arr, y_arr, bracket=None):
    """
    Find zero crossing of y(x) using cubic spline interpolation + Brent method.

    Parameters
    ----------
    x_arr : array-like
        x values (monotone)
    y_arr : array-like
        y values
    bracket : tuple (a, b), optional
        Search bracket; if None, find first sign change

    Returns
    -------
    x_zero : float or None
        Location of zero, or None if not found
    """
    cs = CubicSpline(x_arr, y_arr)

    if bracket is not None:
        a, b = bracket
        try:
            return brentq(cs, a, b, xtol=1e-10, rtol=1e-10)
        except ValueError:
            return None

    # Find sign changes
    signs = np.sign(y_arr)
    sign_changes = np.where(np.diff(signs) != 0)[0]

    if len(sign_changes) == 0:
        return None

    # Use first sign change
    i = sign_changes[0]
    a, b = x_arr[i], x_arr[i + 1]
    try:
        return brentq(cs, a, b, xtol=1e-10, rtol=1e-10)
    except ValueError:
        return None


def running_vev(t, M_planck, higgs_vev):
    """
    Running Higgs VEV: v(t) = M_Pl * exp(-t)
    At t = ln(M_Pl/v), v(t) = v.
    """
    return M_planck * np.exp(-t)


def linear_fit_slope(x, y):
    """Compute slope from linear regression of (x, y) data."""
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if n < 2:
        return np.nan
    slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
        n * np.sum(x ** 2) - np.sum(x) ** 2
    )
    return slope
