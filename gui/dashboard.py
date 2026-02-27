"""
Real-time monitoring panels embedded in the tkinter GUI.

Uses matplotlib FigureCanvasTkAgg for live plot updates.
"""

import numpy as np
import tkinter as tk
from tkinter import ttk

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

from config.parameters import (
    G_STAR, LAMBDA_STAR, THETA_I, HIGGS_MASS_EXP, HIGGS_MASS_THEORY, M_PLANCK
)


class ProgressPanel(ttk.Frame):
    """Top status row: progress bar, phase indicator, step size, ETA."""

    def __init__(self, parent):
        super().__init__(parent, padding=4)
        self._build_ui()

    def _build_ui(self):
        # RG time progress
        ttk.Label(self, text="RG Time:").grid(row=0, column=0, sticky="w", padx=4)
        self.time_label = ttk.Label(self, text="0.00 / 39.34 e-folds", width=20)
        self.time_label.grid(row=0, column=1, sticky="w", padx=4)
        self.progress_bar = ttk.Progressbar(self, length=200, mode="determinate")
        self.progress_bar.grid(row=0, column=2, padx=4)

        # Phase
        ttk.Label(self, text="Phase:").grid(row=0, column=3, sticky="w", padx=4)
        self.phase_label = ttk.Label(self, text="φ = 0.000 rad", width=16)
        self.phase_label.grid(row=0, column=4, sticky="w", padx=4)

        # Step size
        ttk.Label(self, text="Step:").grid(row=0, column=5, sticky="w", padx=4)
        self.step_label = ttk.Label(self, text="Δt = ---", width=14)
        self.step_label.grid(row=0, column=6, sticky="w", padx=4)

        # Status
        ttk.Label(self, text="Status:").grid(row=0, column=7, sticky="w", padx=4)
        self.status_label = ttk.Label(self, text="Idle", foreground="blue", width=12)
        self.status_label.grid(row=0, column=8, sticky="w", padx=4)

    def update(self, t, N_max, step_size=None, status="Running"):
        pct = min(100.0, t / N_max * 100.0)
        self.progress_bar["value"] = pct
        self.time_label.config(text=f"{t:.2f} / {N_max:.2f} e-folds")
        phase = (THETA_I * t) % (2.0 * np.pi)
        self.phase_label.config(text=f"φ = {phase:.3f} rad")
        if step_size is not None:
            self.step_label.config(text=f"Δt = {step_size:.2e}")
        self.status_label.config(text=status)


class SpiralPlotPanel(ttk.Frame):
    """Live-updating spiral trajectory panel."""

    def __init__(self, parent):
        super().__init__(parent)
        if not MATPLOTLIB_AVAILABLE:
            ttk.Label(self, text="matplotlib not available").pack()
            return
        self._build_ui()

    def _build_ui(self):
        self.fig = Figure(figsize=(12, 3.5), dpi=80)
        self.ax_traj = self.fig.add_subplot(131)
        self.ax_G = self.fig.add_subplot(132)
        self.ax_L = self.fig.add_subplot(133)
        self.fig.tight_layout(pad=1.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._init_plots()

    def _init_plots(self):
        self.ax_traj.set_xlabel(r"$\tilde{G}$")
        self.ax_traj.set_ylabel(r"$\tilde{\Lambda}$")
        self.ax_traj.set_title("Coupling Space Trajectory")
        self.ax_traj.plot(G_STAR, LAMBDA_STAR, 'rx', ms=10, mew=2, label="NGFP")
        self.ax_traj.legend(fontsize=7)
        self.ax_traj.grid(True, alpha=0.3)

        self.ax_G.set_xlabel("t (e-folds)")
        self.ax_G.set_ylabel(r"$\tilde{G}$")
        self.ax_G.set_title(r"$\tilde{G}(t)$")
        self.ax_G.axhline(G_STAR, color='r', ls='--', lw=0.8)
        self.ax_G.grid(True, alpha=0.3)

        self.ax_L.set_xlabel("t (e-folds)")
        self.ax_L.set_ylabel(r"$\tilde{\Lambda}$")
        self.ax_L.set_title(r"$\tilde{\Lambda}(t)$")
        self.ax_L.axhline(LAMBDA_STAR, color='r', ls='--', lw=0.8)
        self.ax_L.grid(True, alpha=0.3)

    def update_plots(self, t_arr, G_arr, Lambda_arr):
        if not MATPLOTLIB_AVAILABLE:
            return
        n = len(t_arr)
        colors = plt.cm.coolwarm(np.linspace(0, 1, max(n, 2)))

        self.ax_traj.cla()
        self._init_plots()
        for i in range(max(n - 1, 0)):
            self.ax_traj.plot(G_arr[i:i+2], Lambda_arr[i:i+2],
                               color=colors[i], lw=0.8, alpha=0.7)

        self.ax_G.cla()
        self.ax_G.plot(t_arr, G_arr, 'b-', lw=1)
        self.ax_G.axhline(G_STAR, color='r', ls='--', lw=0.8)
        self.ax_G.set_xlabel("t (e-folds)")
        self.ax_G.set_ylabel(r"$\tilde{G}$")
        self.ax_G.grid(True, alpha=0.3)

        self.ax_L.cla()
        self.ax_L.plot(t_arr, Lambda_arr, 'g-', lw=1)
        self.ax_L.axhline(LAMBDA_STAR, color='r', ls='--', lw=0.8)
        self.ax_L.set_xlabel("t (e-folds)")
        self.ax_L.set_ylabel(r"$\tilde{\Lambda}$")
        self.ax_L.grid(True, alpha=0.3)

        self.canvas.draw_idle()


class MassExtractionPanel(ttk.Frame):
    """Panel for mass extraction and propagator visualization."""

    def __init__(self, parent):
        super().__init__(parent)
        if not MATPLOTLIB_AVAILABLE:
            ttk.Label(self, text="matplotlib not available").pack()
            return
        self._build_ui()

    def _build_ui(self):
        self.fig = Figure(figsize=(12, 3.5), dpi=80)
        self.ax_prop = self.fig.add_subplot(131)
        self.ax_mass = self.fig.add_subplot(132)
        self.ax_vev = self.fig.add_subplot(133)
        self.fig.tight_layout(pad=1.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._init_plots()

    def _init_plots(self):
        self.ax_prop.set_xlabel(r"$\sqrt{-p^2}$ (GeV)")
        self.ax_prop.set_ylabel(r"$\Gamma^{(2)}$ (norm.)")
        self.ax_prop.set_title("Inverse Propagator")
        self.ax_prop.grid(True, alpha=0.3)
        self.ax_prop.axhline(0, color='k', lw=0.8)

        self.ax_mass.set_xlabel("t (e-folds)")
        self.ax_mass.set_ylabel(r"$m_h$ (GeV)")
        self.ax_mass.set_title("Mass Evolution")
        self.ax_mass.axhline(HIGGS_MASS_EXP, color='g', ls='--', lw=1, label="Exp")
        self.ax_mass.axhline(HIGGS_MASS_THEORY, color='r', ls=':', lw=1, label="Theory")
        self.ax_mass.legend(fontsize=7)
        self.ax_mass.set_ylim(110, 140)
        self.ax_mass.grid(True, alpha=0.3)

        self.ax_vev.set_xlabel("t (e-folds)")
        self.ax_vev.set_ylabel("v(t) (GeV)")
        self.ax_vev.set_title("Running VEV")
        self.ax_vev.grid(True, alpha=0.3)

    def update_propagator(self, p2_arr, inv_prop_arr, p2_pole=None):
        if not MATPLOTLIB_AVAILABLE:
            return
        self.ax_prop.cla()
        mass_arr = np.sqrt(np.maximum(-p2_arr, 0))
        norm = np.max(np.abs(inv_prop_arr)) or 1.0
        self.ax_prop.plot(mass_arr, inv_prop_arr / norm, 'b-', lw=1.5)
        self.ax_prop.axhline(0, color='k', lw=0.8)
        if p2_pole is not None and p2_pole < 0:
            m_pole = np.sqrt(-p2_pole)
            self.ax_prop.axvline(m_pole, color='r', ls='--', lw=1.5,
                                  label=f"m_h = {m_pole:.2f} GeV")
            self.ax_prop.legend(fontsize=7)
        self.ax_prop.set_xlabel(r"$\sqrt{-p^2}$ (GeV)")
        self.ax_prop.set_ylabel(r"$\Gamma^{(2)}$ (norm.)")
        self.ax_prop.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def update_mass_evolution(self, t_arr, mass_arr):
        if not MATPLOTLIB_AVAILABLE:
            return
        self.ax_mass.cla()
        valid = ~np.isnan(mass_arr)
        if np.any(valid):
            self.ax_mass.plot(t_arr[valid], mass_arr[valid], 'b-', lw=1)
        self.ax_mass.axhline(HIGGS_MASS_EXP, color='g', ls='--', lw=1, label="Exp")
        self.ax_mass.axhline(HIGGS_MASS_THEORY, color='r', ls=':', lw=1, label="Theory")
        self.ax_mass.legend(fontsize=7)
        self.ax_mass.set_ylim(110, 140)
        self.ax_mass.set_xlabel("t (e-folds)")
        self.ax_mass.set_ylabel(r"$m_h$ (GeV)")
        self.ax_mass.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def update_vev(self, t_arr):
        if not MATPLOTLIB_AVAILABLE:
            return
        self.ax_vev.cla()
        vev = M_PLANCK * np.exp(-t_arr)
        self.ax_vev.semilogy(t_arr, vev, 'purple', lw=1)
        self.ax_vev.set_xlabel("t (e-folds)")
        self.ax_vev.set_ylabel("v(t) (GeV)")
        self.ax_vev.grid(True, alpha=0.3)
        self.canvas.draw_idle()
