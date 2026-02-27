"""
Parameter controls and action buttons for the tkinter GUI.

Provides slider widgets, radio buttons, and action buttons
for the control panel sidebar.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from config.parameters import (
    THETA_R, THETA_I, DEFAULT_N_MAX, DEFAULT_RTOL, DEFAULT_ATOL
)


class ControlPanel(ttk.Frame):
    """
    Sidebar control panel with parameter sliders, radio buttons, and action buttons.
    """

    def __init__(self, parent, on_run=None, on_scan=None, on_convergence=None,
                  on_reset=None, on_export=None, on_load_preset=None):
        super().__init__(parent, padding=8)
        self.on_run = on_run
        self.on_scan = on_scan
        self.on_convergence = on_convergence
        self.on_reset = on_reset
        self.on_export = on_export
        self.on_load_preset = on_load_preset

        # Parameter variables
        self.theta_i_var = tk.DoubleVar(value=THETA_I)
        self.theta_r_var = tk.DoubleVar(value=THETA_R)
        self.N_max_var = tk.DoubleVar(value=DEFAULT_N_MAX)
        self.rtol_var = tk.DoubleVar(value=np.log10(DEFAULT_RTOL))
        self.atol_var = tk.DoubleVar(value=np.log10(DEFAULT_ATOL))
        self.termination_var = tk.StringVar(value="fixed")

        self._build_ui()

    def _build_ui(self):
        row = 0

        # Title
        ttk.Label(self, text="PARAMETERS", font=("Helvetica", 10, "bold")).grid(
            row=row, column=0, columnspan=2, pady=(0, 6), sticky="w"
        )
        row += 1

        # --- Sliders ---
        slider_specs = [
            ("θ_i", self.theta_i_var, 2.0, 2.8, 0.001),
            ("θ_r", self.theta_r_var, 2.0, 3.5, 0.001),
            ("N_max", self.N_max_var, 35.0, 43.0, 0.01),
        ]

        for label, var, lo, hi, res in slider_specs:
            ttk.Label(self, text=f"{label}:").grid(row=row, column=0, sticky="w", pady=2)
            val_label = ttk.Label(self, text=f"{var.get():.3f}", width=8)
            val_label.grid(row=row, column=1, sticky="e")
            row += 1

            def make_update(vl, v):
                def _update(val):
                    vl.config(text=f"{float(val):.4f}")
                return _update

            sl = ttk.Scale(self, from_=lo, to=hi, variable=var,
                           orient="horizontal", length=160,
                           command=make_update(val_label, var))
            sl.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 4))
            row += 1

        # rtol slider (log scale)
        ttk.Label(self, text="log10(rtol):").grid(row=row, column=0, sticky="w", pady=2)
        rtol_label = ttk.Label(self, text=f"{self.rtol_var.get():.1f}", width=8)
        rtol_label.grid(row=row, column=1, sticky="e")
        row += 1
        ttk.Scale(self, from_=-10, to=-6, variable=self.rtol_var, orient="horizontal",
                  length=160,
                  command=lambda v: rtol_label.config(text=f"{float(v):.2f}")).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(0, 4)
        )
        row += 1

        ttk.Label(self, text="log10(atol):").grid(row=row, column=0, sticky="w", pady=2)
        atol_label = ttk.Label(self, text=f"{self.atol_var.get():.1f}", width=8)
        atol_label.grid(row=row, column=1, sticky="e")
        row += 1
        ttk.Scale(self, from_=-12, to=-8, variable=self.atol_var, orient="horizontal",
                  length=160,
                  command=lambda v: atol_label.config(text=f"{float(v):.2f}")).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(0, 8)
        )
        row += 1

        # --- Termination method ---
        ttk.Label(self, text="TERMINATION", font=("Helvetica", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(4, 2)
        )
        row += 1
        for method, text in [("fixed", "Fixed Scale"), ("resonance", "Resonance"),
                              ("variational", "Variational")]:
            ttk.Radiobutton(self, text=text, variable=self.termination_var,
                            value=method).grid(row=row, column=0, columnspan=2, sticky="w")
            row += 1

        ttk.Separator(self, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=8
        )
        row += 1

        # --- Action buttons ---
        ttk.Label(self, text="ACTIONS", font=("Helvetica", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        row += 1

        run_btn = ttk.Button(self, text="RUN INTEGRATION",
                              command=self._on_run_click)
        run_btn.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        row += 1

        ttk.Button(self, text="PARAMETER SCAN",
                   command=self.on_scan).grid(row=row, column=0, columnspan=2,
                                               sticky="ew", pady=2)
        row += 1

        ttk.Button(self, text="CONVERGENCE TEST",
                   command=self.on_convergence).grid(row=row, column=0, columnspan=2,
                                                      sticky="ew", pady=2)
        row += 1

        ttk.Button(self, text="RESET TO DEFAULTS",
                   command=self._on_reset_click).grid(row=row, column=0, columnspan=2,
                                                       sticky="ew", pady=2)
        row += 1

        ttk.Button(self, text="EXPORT DATA",
                   command=self.on_export).grid(row=row, column=0, columnspan=2,
                                                 sticky="ew", pady=2)
        row += 1

    def _on_run_click(self):
        if self.on_run:
            self.on_run(self.get_params())

    def _on_reset_click(self):
        self.theta_i_var.set(THETA_I)
        self.theta_r_var.set(THETA_R)
        self.N_max_var.set(DEFAULT_N_MAX)
        self.rtol_var.set(np.log10(DEFAULT_RTOL))
        self.atol_var.set(np.log10(DEFAULT_ATOL))
        self.termination_var.set("fixed")
        if self.on_reset:
            self.on_reset()

    def get_params(self):
        """Return current parameter values as a dictionary."""
        return {
            "theta_i": self.theta_i_var.get(),
            "theta_r": self.theta_r_var.get(),
            "N_max": self.N_max_var.get(),
            "rtol": 10.0 ** self.rtol_var.get(),
            "atol": 10.0 ** self.atol_var.get(),
            "termination": self.termination_var.get(),
        }
