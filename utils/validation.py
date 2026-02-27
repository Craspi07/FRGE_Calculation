"""Input validation and error handling."""

import numpy as np
import warnings


def validate_parameters(theta_i, theta_r, N_max, rtol, atol):
    """
    Comprehensive parameter validation with physical bounds.

    Raises
    ------
    ValueError
        If parameters are outside acceptable ranges.
    """
    errors = []

    # Physical reasonableness
    if not 1.5 < theta_i < 4.0:
        errors.append(f"theta_i = {theta_i:.4f} outside reasonable range [1.5, 4.0]")

    if not 1.5 < theta_r < 5.0:
        errors.append(f"theta_r = {theta_r:.4f} outside reasonable range [1.5, 5.0]")

    if not 30 < N_max < 45:
        errors.append(f"N_max = {N_max:.4f} outside reasonable range [30, 45]")

    # Numerical stability
    if rtol < atol:
        errors.append(
            f"Relative tolerance rtol={rtol:.2e} should be >= absolute tolerance atol={atol:.2e}"
        )

    if rtol > 1e-4:
        errors.append(f"rtol={rtol:.2e} too loose - results will be inaccurate")

    if atol < 1e-15:
        errors.append(f"atol={atol:.2e} too tight - may cause numerical underflow")

    if errors:
        raise ValueError("Parameter validation failed:\n" + "\n".join(errors))

    # Non-fatal warnings
    T_RG = 2.0 * np.pi / theta_i
    max_step_bound = T_RG / 50.0
    if rtol > 1e-5:
        warnings.warn(
            f"rtol={rtol:.2e} may be marginal for resolving spiral with period {T_RG:.3f}",
            UserWarning,
            stacklevel=2,
        )


def validate_integration_result(sol):
    """
    Monitor integration solution for common failure modes.

    Parameters
    ----------
    sol : OdeResult
        Result from scipy solve_ivp

    Returns
    -------
    bool
        True if valid

    Raises
    ------
    RuntimeError
        If critical failure detected
    """
    if sol is None:
        raise RuntimeError("Integration returned None")

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    if np.any(np.isnan(sol.y)):
        raise RuntimeError("Integration produced NaN values")

    if np.any(np.isinf(sol.y)):
        raise RuntimeError("Integration produced infinite values")

    # Check spiral behavior in gravitational couplings
    if sol.y.shape[0] >= 2 and len(sol.t) >= 200:
        G_traj = sol.y[0, -100:]
        if np.std(G_traj) < 1e-9:
            warnings.warn(
                "Spiral oscillation may have been lost in G coupling",
                UserWarning,
                stacklevel=2,
            )

    # Step size adaptation check
    if len(sol.t) > 2:
        step_sizes = np.diff(sol.t)
        ratio = np.max(step_sizes) / max(np.min(step_sizes), 1e-20)
        if ratio > 100000:
            warnings.warn(
                f"Large step size variation (ratio={ratio:.1f}) - integration may be unstable",
                UserWarning,
                stacklevel=2,
            )

    return True
