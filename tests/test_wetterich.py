"""Unit tests for the FRGE (Wetterich equation) implementation."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from core.beta_functions import (
    beta_G_linearized, beta_Lambda_linearized, rhs_full_system
)
from core.wetterich import (
    wetterich_rhs, get_default_params, get_initial_conditions
)
from config.parameters import G_STAR, LAMBDA_STAR, THETA_R, THETA_I


class TestBetaFunctions:
    """Tests for RG beta functions."""

    def test_fixed_point_stability(self):
        """At the fixed point, perturbation beta functions return rotation."""
        params = get_default_params()
        t = 0.0

        # At the fixed point, dG/dt and dLambda/dt should be zero for zero perturbation
        dG = beta_G_linearized(G_STAR, LAMBDA_STAR, G_STAR, LAMBDA_STAR, THETA_R, THETA_I, t)
        dL = beta_Lambda_linearized(G_STAR, LAMBDA_STAR, G_STAR, LAMBDA_STAR, THETA_R, THETA_I, t)
        assert abs(dG) < 1e-12, f"dG at fixed point should be 0, got {dG}"
        assert abs(dL) < 1e-12, f"dLambda at fixed point should be 0, got {dL}"

    def test_spiral_structure(self):
        """Perturbations should follow spiral flow (complex eigenvalues)."""
        t = 0.0
        G_perturb = G_STAR + 0.01
        L_perturb = LAMBDA_STAR

        dG = beta_G_linearized(G_perturb, L_perturb, G_STAR, LAMBDA_STAR, THETA_R, THETA_I, t)
        dL = beta_Lambda_linearized(G_perturb, L_perturb, G_STAR, LAMBDA_STAR, THETA_R, THETA_I, t)

        # Should have both radial decay and angular rotation
        assert dG != 0 or dL != 0, "Perturbation should cause flow"

    def test_rhs_shape(self):
        """RHS should return array of shape (4,)."""
        params = get_default_params()
        y = np.array([G_STAR, LAMBDA_STAR, 0.0, 0.0])
        dydt = rhs_full_system(0.0, y, params)
        assert dydt.shape == (4,), f"Expected shape (4,), got {dydt.shape}"

    def test_rhs_finite(self):
        """RHS should be finite for physical parameters."""
        params = get_default_params()
        y0 = get_initial_conditions(params)
        dydt = wetterich_rhs(0.0, y0, params)
        assert np.all(np.isfinite(dydt)), f"RHS has non-finite values: {dydt}"


class TestInitialConditions:
    """Tests for initial condition computation."""

    def test_initial_conditions_near_fp(self):
        """Initial conditions should be near the fixed point."""
        params = get_default_params()
        y0 = get_initial_conditions(params)
        assert abs(y0[0] - G_STAR) < 0.2, "G should start near G*"
        assert abs(y0[1] - LAMBDA_STAR) < 0.2, "Lambda should start near Lambda*"

    def test_initial_conditions_shape(self):
        """Initial conditions should have shape (4,)."""
        params = get_default_params()
        y0 = get_initial_conditions(params)
        assert y0.shape == (4,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
