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
