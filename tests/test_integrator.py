"""Integration accuracy tests for the FRGE flow integrator."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from core.integrator import run_integration
from core.spiral import count_spiral_windings, phase_wrapping_times
from config.parameters import THETA_R, THETA_I, DEFAULT_N_MAX, G_STAR, LAMBDA_STAR


class TestIntegrator:
    """Tests for the RG flow integrator."""

    @pytest.fixture
    def fast_result(self):
        """Run a fast integration for testing (loose tolerances)."""
        return run_integration(rtol=1e-5, atol=1e-7)

    def test_integration_success(self, fast_result):
        """Integration should complete successfully."""
        assert fast_result["success"], f"Integration failed: {fast_result.get('message')}"

    def test_integration_shape(self, fast_result):
        """Output arrays should have correct shape."""
        assert fast_result["y"].shape[0] == 4, "y should have 4 rows (G, L, lam_h, m2_h)"
        assert len(fast_result["t"]) > 10, "Should have at least 10 time steps"
        assert fast_result["y"].shape[1] == len(fast_result["t"]), "y and t shapes should match"

    def test_no_nan_inf(self, fast_result):
        """Integration should not produce NaN or Inf."""
        assert np.all(np.isfinite(fast_result["y"])), "Integration has NaN or Inf values"

    def test_approaches_fixed_point(self, fast_result):
        """G and Lambda should approach the fixed point as t increases."""
        G_final = fast_result["y"][0, -1]
        L_final = fast_result["y"][1, -1]

        G_init = fast_result["y"][0, 0]
        L_init = fast_result["y"][1, 0]

        dist_final = np.sqrt((G_final - G_STAR)**2 + (L_final - LAMBDA_STAR)**2)
        dist_init = np.sqrt((G_init - G_STAR)**2 + (L_init - LAMBDA_STAR)**2)

        assert dist_final < dist_init, (
            f"Should approach fixed point: init dist={dist_init:.4f}, "
            f"final dist={dist_final:.4f}"
        )

    def test_termination_options(self):
        """All termination methods should succeed."""
        for method in ["fixed", "resonance"]:
            result = run_integration(rtol=1e-5, atol=1e-7, termination=method)
            assert result["success"], f"Termination method '{method}' failed"

    def test_spiral_windings(self, fast_result):
        """Should complete approximately 15 spiral windings."""
        N_max = fast_result["statistics"]["N_max"]
        windings = count_spiral_windings(N_max, THETA_I)
        assert 14 < windings < 17, f"Expected ~15 windings, got {windings:.2f}"

    def test_step_statistics(self, fast_result):
        """Step statistics should be reasonable."""
        stats = fast_result["statistics"]
        assert stats["n_steps"] > 100, "Should have more than 100 steps"
        assert stats["avg_step"] > 0, "Average step should be positive"
        assert stats["min_step"] > 0, "Min step should be positive"

    def test_g_tilde_range(self, fast_result):
        """G~ should remain in physically reasonable range."""
        G_arr = fast_result["y"][0]
        assert np.all(G_arr > 0), "G~ should remain positive"
        assert np.all(G_arr < 2.0), "G~ should not diverge"


class TestPhaseWrapping:
    """Tests for spiral phase tracking."""

    def test_wrapping_times(self):
        """Phase wrapping times should be at 2*pi*n/theta_i."""
        wraps = phase_wrapping_times(DEFAULT_N_MAX, THETA_I)
        T = 2.0 * np.pi / THETA_I
        for i, tw in enumerate(wraps):
            expected = T * (i + 1)
            assert abs(tw - expected) < 1e-10, f"Wrap time {i+1} should be {expected:.4f}, got {tw:.4f}"

    def test_wrapping_count(self):
        """Should have ~15 wrapping events in N_max."""
        wraps = phase_wrapping_times(DEFAULT_N_MAX, THETA_I)
        assert 13 < len(wraps) < 18, f"Expected ~15 wrapping events, got {len(wraps)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
