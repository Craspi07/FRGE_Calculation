"""
Interactive plotting utilities using matplotlib.

Provides reusable plot functions for the dashboard and results panels.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from config.parameters import (
    G_STAR, LAMBDA_STAR, THETA_I, HIGGS_MASS_EXP, HIGGS_MASS_THEORY,
    M_PLANCK
)
from core.spiral import distance_to_fixed_point, phase_wrapping_times
from utils.math_utils import linear_fit_slope


def plot_spiral_trajectory(t_arr, G_arr, Lambda_arr, ax=None, title=True):
    """
    Plot 2D spiral trajectory in (G~, Lambda~) coupling space.
    Color-coded from blue (early) to red (late).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    n = len(t_arr)
    colors = plt.cm.coolwarm(np.linspace(0, 1, n))

    for i in range(n - 1):
        ax.plot(G_arr[i:i+2], Lambda_arr[i:i+2], color=colors[i], lw=0.8, alpha=0.7)

    # Fixed point
    ax.plot(G_STAR, LAMBDA_STAR, 'rx', markersize=12, markeredgewidth=2,
            label=f'NGFP ({G_STAR}, {LAMBDA_STAR})', zorder=5)

    # Start/end markers
    ax.plot(G_arr[0], Lambda_arr[0], 'go', markersize=6, label='t=0 (Planck)')
    ax.plot(G_arr[-1], Lambda_arr[-1], 'rs', markersize=6, label='t=N_max (EW)')

    ax.set_xlabel(r'$\tilde{G}$', fontsize=11)
    ax.set_ylabel(r'$\tilde{\Lambda}$', fontsize=11)
    if title:
        ax.set_title('Spiral Trajectory in Coupling Space', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_coupling_evolution(t_arr, G_arr, Lambda_arr, axes=None):
    """Plot G~(t) and Lambda~(t) time series."""
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    else:
        fig = axes[0].figure

    axes[0].plot(t_arr, G_arr, 'b-', lw=1.2, label=r'$\tilde{G}(t)$')
    axes[0].axhline(G_STAR, color='r', ls='--', lw=1, alpha=0.6, label=f'G* = {G_STAR}')
    axes[0].set_xlabel('RG time t (e-folds)', fontsize=10)
    axes[0].set_ylabel(r'$\tilde{G}$', fontsize=10)
    axes[0].set_title(r'Newton Coupling $\tilde{G}(t)$', fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_arr, Lambda_arr, 'g-', lw=1.2, label=r'$\tilde{\Lambda}(t)$')
    axes[1].axhline(LAMBDA_STAR, color='r', ls='--', lw=1, alpha=0.6,
                    label=f'L* = {LAMBDA_STAR}')
    axes[1].set_xlabel('RG time t (e-folds)', fontsize=10)
    axes[1].set_ylabel(r'$\tilde{\Lambda}$', fontsize=10)
    axes[1].set_title(r'Cosmological Constant $\tilde{\Lambda}(t)$', fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    return fig, axes


def plot_distance_to_fixed_point(t_arr, G_arr, Lambda_arr, theta_r, ax=None):
    """
    Plot log(distance to fixed point) vs t.
    Should be linear with slope -theta_r.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    dist = distance_to_fixed_point(G_arr, Lambda_arr)
    log_dist = np.log(np.maximum(dist, 1e-20))

    ax.plot(t_arr, log_dist, 'b-', lw=1.2, label='log d(t)')

    # Fit slope
    mask = (t_arr > 1) & (log_dist > -40)
    if np.sum(mask) > 5:
        slope = linear_fit_slope(t_arr[mask], log_dist[mask])
        ax.plot(t_arr, log_dist[mask][0] + slope * (t_arr - t_arr[mask][0]),
                'r--', lw=1, alpha=0.8, label=f'Fit slope = {slope:.3f}\n(expected: {-theta_r:.3f})')

    # Phase wrapping markers
    wraps = phase_wrapping_times(t_arr[-1], THETA_I)
    for tw in wraps:
        ax.axvline(tw, color='gray', ls=':', alpha=0.5, lw=0.8)

    ax.set_xlabel('RG time t (e-folds)', fontsize=10)
    ax.set_ylabel('log d(t)', fontsize=10)
    ax.set_title('Distance to Fixed Point (log scale)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_inverse_propagator(p2_arr, inv_prop_arr, p2_pole=None, ax=None):
    """
    Plot inverse propagator Gamma^(2)(p^2) near the pole.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    # Convert p^2 to GeV (mass units): x = sqrt(-p^2)
    mass_arr = np.sqrt(-p2_arr)

    ax.plot(mass_arr, inv_prop_arr / np.max(np.abs(inv_prop_arr)), 'b-', lw=1.5,
            label=r'$\Gamma^{(2)}(p^2)$ (normalized)')
    ax.axhline(0, color='k', ls='-', lw=0.8)

    if p2_pole is not None and p2_pole < 0:
        m_pole = np.sqrt(-p2_pole)
        ax.axvline(m_pole, color='r', ls='--', lw=1.5,
                   label=f'Pole: m_h = {m_pole:.2f} GeV')

    ax.set_xlabel(r'$\sqrt{-p^2}$ (GeV)', fontsize=10)
    ax.set_ylabel(r'$\Gamma^{(2)}$ (normalized)', fontsize=10)
    ax.set_title('Inverse Propagator near Higgs Pole', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_mass_evolution(t_arr, mass_arr, ax=None):
    """
    Plot extracted mass evolution m_h(t) during integration.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    valid = ~np.isnan(mass_arr)
    if np.any(valid):
        ax.plot(t_arr[valid], mass_arr[valid], 'b-', lw=1.2, label=r'$m_h(t)$')

    ax.axhline(HIGGS_MASS_EXP, color='g', ls='--', lw=1.5,
               label=f'Experiment: {HIGGS_MASS_EXP:.2f} GeV')
    ax.axhline(HIGGS_MASS_THEORY, color='r', ls=':', lw=1.5,
               label=f'Theory: {HIGGS_MASS_THEORY:.2f} GeV')

    ax.set_xlabel('RG time t (e-folds)', fontsize=10)
    ax.set_ylabel(r'$m_h$ (GeV)', fontsize=10)
    ax.set_title('Higgs Mass Evolution', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(115, 135)

    return fig, ax


def plot_running_vev(t_arr, ax=None):
    """Plot running Higgs VEV v(t) = M_Pl * exp(-t)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    vev = M_PLANCK * np.exp(-t_arr)
    ax.semilogy(t_arr, vev, 'purple', lw=1.5, label='v(t) = M_Pl exp(-t)')
    ax.set_xlabel('RG time t (e-folds)', fontsize=10)
    ax.set_ylabel('v(t) (GeV)', fontsize=10)
    ax.set_title('Running Higgs VEV', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_parameter_scan_heatmap(theta_i_arr, N_max_arr, mass_grid, ax=None,
                                  sweet_spot=None, current_point=None):
    """
    2D contour heatmap of extracted Higgs mass over (theta_i, N_max) space.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    TI, NM = np.meshgrid(theta_i_arr, N_max_arr, indexing='ij')
    valid = np.isfinite(mass_grid)
    if not np.any(valid):
        ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes, ha='center')
        return fig, ax

    vmin = np.nanpercentile(mass_grid, 5)
    vmax = np.nanpercentile(mass_grid, 95)

    cf = ax.contourf(TI, NM, mass_grid, levels=30, cmap='RdYlGn',
                     vmin=vmin, vmax=vmax)
    plt.colorbar(cf, ax=ax, label='m_h (GeV)')

    # Contour lines at specific values
    for m_target in [124.0, 125.0, 125.25, 126.0, 127.0]:
        if vmin < m_target < vmax:
            cs = ax.contour(TI, NM, mass_grid, levels=[m_target],
                            colors=['black'], linewidths=1)
            ax.clabel(cs, fmt=f'{m_target:.1f}', fontsize=7)

    if sweet_spot:
        ax.plot(sweet_spot["theta_i"], sweet_spot["N_max"], 'w*',
                markersize=14, label=f"Sweet spot\nm_h={sweet_spot['m_h']:.2f} GeV")

    if current_point:
        ax.plot(current_point[0], current_point[1], 'ko', markersize=8,
                label='Current params')

    ax.set_xlabel(r'$\theta_i$', fontsize=11)
    ax.set_ylabel(r'$N_{max}$ (e-folds)', fontsize=11)
    ax.set_title('Higgs Mass Parameter Scan', fontsize=10)
    ax.legend(fontsize=7, loc='upper right')

    return fig, ax


def create_full_dashboard(integration_result, pole_result=None):
    """
    Create a complete dashboard figure with all panels.
    Returns a matplotlib Figure.
    """
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Higgs Propagator Calculation - Reuter Spiral Background',
                 fontsize=14, fontweight='bold')

    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    t_arr = integration_result["t"]
    y_arr = integration_result["y"]
    params = integration_result.get("params", {})
    theta_r = params.get("theta_r", 2.714)

    G_arr = y_arr[0]
    Lambda_arr = y_arr[1]

    # Row 1: Trajectory
    ax_traj = fig.add_subplot(gs[0, 0])
    plot_spiral_trajectory(t_arr, G_arr, Lambda_arr, ax=ax_traj, title=True)

    ax_G = fig.add_subplot(gs[0, 1])
    ax_L = fig.add_subplot(gs[0, 2])
    plot_coupling_evolution(t_arr, G_arr, Lambda_arr, axes=[ax_G, ax_L])

    # Row 2: Diagnostics
    ax_dist = fig.add_subplot(gs[1, 0])
    plot_distance_to_fixed_point(t_arr, G_arr, Lambda_arr, theta_r, ax=ax_dist)

    # Step size history
    ax_step = fig.add_subplot(gs[1, 1])
    step_sizes = np.diff(t_arr)
    ax_step.semilogy(t_arr[1:], step_sizes, 'b-', lw=0.8, alpha=0.7)
    ax_step.set_xlabel('RG time t (e-folds)', fontsize=10)
    ax_step.set_ylabel('Step size Δt', fontsize=10)
    ax_step.set_title('Adaptive Step Size History', fontsize=10)
    ax_step.grid(True, alpha=0.3)

    # Phase
    ax_phase = fig.add_subplot(gs[1, 2])
    phase_arr = (THETA_I * t_arr) % (2.0 * np.pi)
    ax_phase.plot(t_arr, phase_arr, 'g-', lw=0.8)
    ax_phase.set_xlabel('RG time t (e-folds)', fontsize=10)
    ax_phase.set_ylabel('Phase φ (rad)', fontsize=10)
    ax_phase.set_title('Spiral Phase mod 2π', fontsize=10)
    ax_phase.grid(True, alpha=0.3)

    # Row 3: Mass extraction
    if pole_result and pole_result.get("success"):
        ax_prop = fig.add_subplot(gs[2, 0])
        plot_inverse_propagator(
            pole_result["p2_arr"], pole_result["inv_prop_arr"],
            p2_pole=pole_result.get("p2_pole"), ax=ax_prop
        )
    else:
        ax_prop = fig.add_subplot(gs[2, 0])
        ax_prop.text(0.5, 0.5, 'No pole found', transform=ax_prop.transAxes,
                     ha='center', va='center', fontsize=12, color='red')

    ax_vev = fig.add_subplot(gs[2, 1])
    plot_running_vev(t_arr, ax=ax_vev)

    # Summary text
    ax_text = fig.add_subplot(gs[2, 2])
    ax_text.axis('off')
    if pole_result and pole_result.get("success"):
        m_h = pole_result.get("m_h", float("nan"))
        summary = (
            f"RESULT SUMMARY\n"
            f"{'='*28}\n"
            f"m_h = {m_h:.3f} GeV\n"
            f"Theory: {HIGGS_MASS_THEORY:.2f} GeV\n"
            f"Exp:    {HIGGS_MASS_EXP:.2f} GeV\n"
            f"Agree:  {abs(m_h - HIGGS_MASS_EXP):.3f} GeV\n\n"
            f"Steps: {integration_result['statistics'].get('n_steps', 0):,}\n"
            f"Time:  {integration_result['statistics'].get('cpu_time', 0):.1f} s"
        )
        color = 'green' if abs(m_h - HIGGS_MASS_EXP) < 0.5 else 'orange'
    else:
        summary = "Integration complete.\nPole extraction failed."
        color = 'red'

    ax_text.text(0.05, 0.95, summary, transform=ax_text.transAxes,
                 va='top', fontsize=9, fontfamily='monospace', color=color)

    return fig


# =============================================================================
# 5 Diagnostic plots  (Part 5 of specification)
# =============================================================================

def plot_3d_spiral(t_arr, G_arr, Lambda_arr, ax=None):
    """
    Plot 1: 3-D spiral trajectory (t, G~, Λ~) — the spiral winding toward NGFP.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers projection)
    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    n = len(t_arr)
    colors = plt.cm.coolwarm(np.linspace(0, 1, n))
    for i in range(n - 1):
        ax.plot(t_arr[i:i+2], G_arr[i:i+2], Lambda_arr[i:i+2],
                color=colors[i], lw=0.7, alpha=0.8)

    ax.scatter([t_arr[-1]], [G_STAR], [LAMBDA_STAR],
               color='red', s=60, marker='x', zorder=5, label='NGFP')
    ax.set_xlabel('t (e-folds)', fontsize=9)
    ax.set_ylabel(r'$\tilde{G}$', fontsize=9)
    ax.set_zlabel(r'$\tilde{\Lambda}$', fontsize=9)
    ax.set_title('3-D Spiral Trajectory', fontsize=10)
    ax.legend(fontsize=8)
    return fig, ax


def plot_phase_portrait(t_arr, G_arr, Lambda_arr, params=None, ax=None):
    """
    Plot 2: Phase portrait — velocity arrows in (G~, Λ~) coupling space.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    theta_r = (params or {}).get("theta_r", 2.714)
    theta_i = (params or {}).get("theta_i", 2.396)

    # Trajectory
    ax.plot(G_arr, Lambda_arr, 'b-', lw=0.8, alpha=0.6, label='Trajectory')
    ax.plot(G_STAR, LAMBDA_STAR, 'rx', ms=12, mew=2, label='NGFP')

    # Velocity field on a coarse grid
    G_lo, G_hi = G_STAR - 0.15, G_STAR + 0.15
    L_lo, L_hi = LAMBDA_STAR - 0.10, LAMBDA_STAR + 0.10
    Gg, Lg = np.meshgrid(np.linspace(G_lo, G_hi, 12),
                          np.linspace(L_lo, L_hi, 12))
    dG  = -theta_r * (Gg - G_STAR) + theta_i * (Lg - LAMBDA_STAR)
    dLg = -theta_i * (Gg - G_STAR) - theta_r * (Lg - LAMBDA_STAR)
    norm = np.sqrt(dG**2 + dLg**2) + 1e-12
    ax.quiver(Gg, Lg, dG / norm, dLg / norm,
              alpha=0.35, color='gray', scale=20, width=0.003)

    ax.set_xlabel(r'$\tilde{G}$', fontsize=11)
    ax.set_ylabel(r'$\tilde{\Lambda}$', fontsize=11)
    ax.set_title('Phase Portrait in Coupling Space', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_mixing_evolution(mixing_result, ax=None):
    """
    Plot 3: Off-diagonal Hessian mixing ratio vs RG time.
    Expects a result dict from analyze_mixing_term().
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    from config.parameters import MIXING_RATIO_TARGET, MIXING_RATIO_MIN, MIXING_RATIO_MAX

    t_arr = mixing_result.get("t_arr", np.array([]))
    mix   = mixing_result.get("mixing_evolution", np.array([]))

    if len(t_arr) > 0:
        ax.plot(t_arr, mix, 'b-', lw=1.2, label=r'$|\Gamma_{h\omega}|^2 / (\Gamma_{hh}\Gamma_{\omega\omega})$')

    ax.axhline(MIXING_RATIO_TARGET, color='r',  ls='--', lw=1.2,
               label=f'Target {MIXING_RATIO_TARGET:.3f}')
    ax.axhspan(MIXING_RATIO_MIN, MIXING_RATIO_MAX, alpha=0.12, color='green',
               label=f'Tolerance [{MIXING_RATIO_MIN:.3f}, {MIXING_RATIO_MAX:.3f}]')

    final = mixing_result.get("final_mixing", float("nan"))
    ok    = mixing_result.get("within_tolerance", False)
    ax.set_xlabel('RG time t (e-folds)', fontsize=10)
    ax.set_ylabel('Mixing ratio', fontsize=10)
    ax.set_title(f'Off-diagonal Hessian Mixing Evolution\n'
                 f'Final = {final:.4f}  {"[PASS]" if ok else "[FAIL]"}', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_convergence_levels(convergence_result, ax=None):
    """
    Plot 4: Extracted mass vs tolerance level (convergence test).
    Expects a result dict from run_convergence_test().
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    from config.parameters import (HIGGS_MASS_EXP, HIGGS_MASS_THEORY,
                                    HIGGS_MASS_TARGET_MIN, HIGGS_MASS_TARGET_MAX)

    details = convergence_result.get("details", [])
    if not details:
        ax.text(0.5, 0.5, 'No convergence data', transform=ax.transAxes,
                ha='center', va='center')
        return fig, ax

    rtols  = [d["rtol"]  for d in details]
    masses = [d["m_h"]   for d in details]
    ok     = [d["success"] for d in details]

    x = np.arange(len(rtols))
    colors_bar = ['steelblue' if s else 'salmon' for s in ok]
    ax.bar(x, masses, color=colors_bar, alpha=0.7, width=0.6)
    ax.axhline(HIGGS_MASS_EXP,    color='g', ls='--', lw=1.5, label=f'Exp {HIGGS_MASS_EXP:.2f} GeV')
    ax.axhline(HIGGS_MASS_THEORY, color='r', ls=':',  lw=1.5, label=f'Theory {HIGGS_MASS_THEORY:.2f} GeV')
    ax.axhspan(HIGGS_MASS_TARGET_MIN, HIGGS_MASS_TARGET_MAX, alpha=0.1, color='green',
               label=f'Confirmed [{HIGGS_MASS_TARGET_MIN}, {HIGGS_MASS_TARGET_MAX}]')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{r:.0e}' for r in rtols], rotation=30, ha='right', fontsize=8)
    ax.set_xlabel('rtol', fontsize=10)
    ax.set_ylabel(r'$m_h$ (GeV)', fontsize=10)
    status = convergence_result.get("status", "")
    ax.set_title(f'Convergence Test — {status}', fontsize=10)
    ax.set_ylim(123, 128)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    return fig, ax


def plot_s2_verification(sphere_result, ax=None):
    """
    Plot 5: S² Monte Carlo verification — distribution of |∇ω|² samples.
    Expects a result dict from verify_sphere_enhancement().
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    from config.parameters import SPHERE_RATIO_TARGET, SPHERE_RATIO_MIN, SPHERE_RATIO_MAX

    ratio  = sphere_result.get("ratio",    float("nan"))
    s2_val = sphere_result.get("s2_value", float("nan"))
    s1_val = sphere_result.get("s1_value", float("nan"))
    n      = sphere_result.get("samples",  0)
    ok     = sphere_result.get("within_tolerance", False)

    labels  = [r'$S^2$ (full sphere)', r'$S^1$ (great circle)']
    heights = [s2_val, s1_val]
    colors  = ['steelblue', 'coral']
    ax.bar(labels, heights, color=colors, alpha=0.75, width=0.5)

    ax.axhline(SPHERE_RATIO_TARGET * s1_val, color='green', ls='--', lw=1.5,
               label=f'4/3 × S¹ = {SPHERE_RATIO_TARGET * s1_val:.4f}')

    for i, h in enumerate(heights):
        ax.text(i, h + 0.005, f'{h:.4f}', ha='center', fontsize=9)

    ax.set_ylabel(r'$\langle|\nabla\omega|^2\rangle$', fontsize=11)
    ax.set_title(
        f'S² Enhancement Verification  (n={n:,})\n'
        f'Ratio = {ratio:.4f}  (target {SPHERE_RATIO_TARGET:.4f} ± 0.05)  '
        f'{"[PASS]" if ok else "[FAIL]"}',
        fontsize=10
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    return fig, ax


# =============================================================================
# Additional scan plots (full-scan mode)
# =============================================================================

def plot_sensitivity_curves(sensitivity_result, axes=None):
    """
    Plot 1-D sensitivity curves: m_h vs each parameter independently.
    Returns a Figure with one subplot per parameter.
    """
    sweeps = sensitivity_result.get("sweeps", {})
    n = len(sweeps)
    if n == 0:
        return plt.figure(), []

    ncols = 3
    nrows = (n + ncols - 1) // ncols
    if axes is None:
        fig, axes_flat = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes_flat = np.asarray(axes_flat).flatten()
    else:
        fig = axes[0].figure
        axes_flat = np.asarray(axes).flatten()

    from config.parameters import HIGGS_MASS_EXP, HIGGS_MASS_THEORY, HIGGS_MASS_TARGET_MIN, HIGGS_MASS_TARGET_MAX

    for idx, (param_name, data) in enumerate(sweeps.items()):
        ax = axes_flat[idx]
        values  = data["values"]
        masses  = data["masses"]
        nominal = data["nominal"]
        label   = data["label"]

        valid = ~np.isnan(masses)
        if np.any(valid):
            ax.plot(values[valid], masses[valid], 'b-o', ms=4, lw=1.2)

        ax.axvline(nominal, color='orange', ls='--', lw=1.2, label=f'nominal = {nominal:.4f}')
        ax.axhline(HIGGS_MASS_EXP,    color='g', ls='--', lw=1, alpha=0.7)
        ax.axhline(HIGGS_MASS_THEORY, color='r', ls=':',  lw=1, alpha=0.7)
        ax.axhspan(HIGGS_MASS_TARGET_MIN, HIGGS_MASS_TARGET_MAX, alpha=0.08, color='green')

        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel(r'$m_h$ (GeV)', fontsize=9)
        ax.set_title(f'Sensitivity: {param_name}', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(len(sweeps), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle('1-D Parameter Sensitivity — m_h vs Each Parameter', fontsize=12)
    fig.tight_layout()
    return fig, axes_flat


def plot_3d_scan_projections(grid_result, figsize=(16, 5)):
    """
    Three 2-D heatmap projections of the 3-D (θ_r × θ_i × N_max) scan.
    """
    proj = grid_result.get("projections", {})
    theta_r_arr = grid_result["theta_r_arr"]
    theta_i_arr = grid_result["theta_i_arr"]
    N_max_arr   = grid_result["N_max_arr"]
    sweet       = grid_result.get("sweet_spot", {})

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    datasets = [
        (proj.get("theta_r_vs_theta_i"), theta_r_arr, theta_i_arr,
         r'$\theta_r$', r'$\theta_i$', "θ_r vs θ_i (mean over N_max)"),
        (proj.get("theta_r_vs_N_max"),   theta_r_arr, N_max_arr,
         r'$\theta_r$', r'$N_{\max}$', "θ_r vs N_max (mean over θ_i)"),
        (proj.get("theta_i_vs_N_max"),   theta_i_arr, N_max_arr,
         r'$\theta_i$', r'$N_{\max}$', "θ_i vs N_max (mean over θ_r)"),
    ]

    from config.parameters import HIGGS_MASS_EXP, HIGGS_MASS_TARGET_MIN, HIGGS_MASS_TARGET_MAX

    for ax, (grid, x_arr, y_arr, xlabel, ylabel, title) in zip(axes, datasets):
        if grid is None:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue
        X, Y = np.meshgrid(x_arr, y_arr, indexing='ij')
        vmin = np.nanpercentile(grid, 5)
        vmax = np.nanpercentile(grid, 95)
        cf = ax.contourf(X, Y, grid, levels=25, cmap='RdYlGn', vmin=vmin, vmax=vmax)
        plt.colorbar(cf, ax=ax, label=r'$m_h$ (GeV)')
        for m_t in [HIGGS_MASS_TARGET_MIN, HIGGS_MASS_EXP, HIGGS_MASS_TARGET_MAX]:
            if vmin < m_t < vmax:
                cs = ax.contour(X, Y, grid, levels=[m_t], colors='black', linewidths=0.8)
                ax.clabel(cs, fmt=f'{m_t:.1f}', fontsize=7)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=9)

    fig.suptitle('3-D Grid Scan Projections — m_h (GeV)', fontsize=12)
    fig.tight_layout()
    return fig, axes


# =============================================================================
# Convenience: save all 5 diagnostic plots in one call
# =============================================================================

def save_all_diagnostic_plots(integration_result, pole_result,
                               mixing_result=None, sphere_result=None,
                               convergence_result=None):
    """
    Generate and save all 5 spec-required diagnostic plots to output/plots/.

    Returns
    -------
    list of str  – paths of saved PNG files
    """
    from utils.export import save_figure_png

    t_arr      = integration_result["t"]
    y_arr      = integration_result["y"]
    params     = integration_result.get("params", {})
    G_arr      = y_arr[0]
    Lambda_arr = y_arr[1]

    saved = []

    # Plot 1 — 3-D spiral
    try:
        fig, _ = plot_3d_spiral(t_arr, G_arr, Lambda_arr)
        saved.append(save_figure_png(fig, "diag_01_3d_spiral.png"))
        plt.close(fig)
    except Exception:
        pass

    # Plot 2 — phase portrait
    try:
        fig, _ = plot_phase_portrait(t_arr, G_arr, Lambda_arr, params=params)
        saved.append(save_figure_png(fig, "diag_02_phase_portrait.png"))
        plt.close(fig)
    except Exception:
        pass

    # Plot 3 — mixing evolution
    if mixing_result:
        try:
            fig, _ = plot_mixing_evolution(mixing_result)
            saved.append(save_figure_png(fig, "diag_03_mixing_evolution.png"))
            plt.close(fig)
        except Exception:
            pass

    # Plot 4 — convergence levels
    if convergence_result:
        try:
            fig, _ = plot_convergence_levels(convergence_result)
            saved.append(save_figure_png(fig, "diag_04_convergence_levels.png"))
            plt.close(fig)
        except Exception:
            pass

    # Plot 5 — S² verification
    if sphere_result:
        try:
            fig, _ = plot_s2_verification(sphere_result)
            saved.append(save_figure_png(fig, "diag_05_s2_verification.png"))
            plt.close(fig)
        except Exception:
            pass

    return saved
