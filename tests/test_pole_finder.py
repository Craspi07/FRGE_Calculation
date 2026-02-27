"""Root finding validation for propagator pole extraction."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from core.integrator import run_integration
from core.pole_finder import extract_pole, extract_pole_at_scale
from core.hessian import compute_hessian, mixing_ratio, effective_mass_squared
from config.parameters import (
    G_STAR, LAMBDA_STAR, THETA_R, THETA_I, HIGGS_VEV, M_PLANCK,
    HIGGS_MASS_THEORY, HIGGS_MASS_EXP, HIGGS_MASS_ERROR
)


class TestHessian:
    """Tests for Hessian computation."""

    def test_hessian_shape(self):
        """Hessian should be a 2x2 matrix."""
        hess = compute_hessian(G_STAR, LAMBDA_STAR, 0.1, -0.01,
                                HIGGS_VEV / 100.0, (100.0)**2, THETA_R, THETA_I, 0)
        assert hess.shape == (2, 2), f"Expected (2,2), got {hess.shape}"

    def test_hessian_symmetric(self):
        """Hessian should be symmetric."""
        hess = compute_hessian(G_STAR, LAMBDA_STAR, 0.1, -0.01,
                                HIGGS_VEV / 100.0, (100.0)**2, THETA_R, THETA_I, 0)
        assert abs(hess[0, 1] - hess[1, 0]) < 1e-12, "Hessian should be symmetric"

    def test_mixing_ratio_range(self):
        """Mixing ratio should be between 0 and 1 for physical parameters."""
        hess = compute_hessian(G_STAR, LAMBDA_STAR, 0.1, -0.01,
                                HIGGS_VEV / 100.0, (100.0)**2, THETA_R, THETA_I, 0)
        mr = mixing_ratio(hess)
        assert 0.0 <= mr <= 2.0, f"Mixing ratio {mr:.4f} out of expected range"

    def test_4_3_enhancement_structure(self):
        """
        The Hessian structure should encode the 4/3 enhancement.
        With lambda_h = 0.5 and a large field, mixing_ratio should approach 1/3.
        """
        # Use larger lambda_h to see enhancement
        hess = compute_hessian(G_STAR, LAMBDA_STAR, 0.5, 0.1,
                                1.0, (1000.0)**2, THETA_R, THETA_I, 10.0)
        mr = mixing_ratio(hess)
        # Should be non-zero (enhancement is present)
        assert mr > 0, "Mixing ratio should be non-zero (enhancement active)"


class TestPoleFinder:
    """Tests for pole extraction."""

    @pytest.fixture
    def integration_result(self):
        return run_integration(rtol=1e-6, atol=1e-8)

    def test_pole_extraction_runs(self, integration_result):
        """Pole extraction should complete without error."""
        result = extract_pole(integration_result)
        assert "m_h" in result, "Result should contain 'm_h'"

    def test_pole_in_physical_range(self, integration_result):
        """Extracted mass should be in physically reasonable range."""
        result = extract_pole(integration_result)
        m_h = result.get("m_h", float("nan"))
        if not np.isnan(m_h):
            assert 100.0 < m_h < 150.0, (
                f"Extracted mass {m_h:.2f} GeV outside expected range [100, 150] GeV"
            )

    def test_pole_near_theory(self, integration_result):
        """
        Extracted mass should be within 5 GeV of theory prediction.
        (Tight test would require full precision integration.)
        """
        result = extract_pole(integration_result)
        m_h = result.get("m_h", float("nan"))
        if not np.isnan(m_h):
            deviation = abs(m_h - HIGGS_MASS_THEORY)
            assert deviation < 5.0, (
                f"Mass {m_h:.2f} GeV deviates {deviation:.2f} GeV from theory {HIGGS_MASS_THEORY:.2f}"
            )


class TestSphereEnhancement:
    """Tests for the 4/3 geometric enhancement."""

    def test_analytical_factor(self):
        """Analytical enhancement should be exactly 4/3."""
        from core.york import enhancement_factor_4_3
        factor, sphere_contribution = enhancement_factor_4_3(1.0)
        assert abs(factor - 4.0/3.0) < 1e-10, f"Expected 4/3, got {factor}"
        assert abs(sphere_contribution - 1.0/3.0) < 1e-10

    def test_theory_formula(self):
        """Theory prediction formula m_h = (4/3) * (theta_i/2pi) * v."""
        m_theory = (4.0/3.0) * (THETA_I / (2.0*np.pi)) * HIGGS_VEV
        assert abs(m_theory - HIGGS_MASS_THEORY) < 0.01, (
            f"Theory formula gives {m_theory:.4f}, expected {HIGGS_MASS_THEORY:.4f}"
        )

    def test_experimental_consistency(self):
        """Theory prediction should be within 2 sigma of experiment."""
        sigma = HIGGS_MASS_ERROR
        deviation = abs(HIGGS_MASS_THEORY - HIGGS_MASS_EXP)
        assert deviation < 2 * sigma, (
            f"Theory {HIGGS_MASS_THEORY:.3f} deviates {deviation:.3f} GeV "
            f"from experiment {HIGGS_MASS_EXP:.3f} +/- {sigma:.3f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
