"""
Primary GUI layout for the Higgs propagator calculation.

Uses tkinter as the main framework with embedded matplotlib figures.
Launches a comprehensive dashboard with real-time monitoring.
"""

import sys
import os
import logging
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np

try:
    import matplotlib
    matplotlib.use("TkAgg")
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from config.parameters import (
    DEFAULT_N_MAX, THETA_R, THETA_I, DEFAULT_RTOL, DEFAULT_ATOL
)
from core.integrator import run_integration
from core.pole_finder import extract_pole
from diagnostics.convergence import run_convergence_test
from diagnostics.mixing_analysis import analyze_mixing_term
from diagnostics.sphere_integral import verify_sphere_enhancement
from diagnostics.phase_sensitivity import check_phase_sensitivity
from diagnostics.parameter_scan import run_parameter_scan
from utils.export import (
    export_results_json, format_results_summary,
    output_path, save_figure_png, setup_output_dirs,
)

logger = logging.getLogger(__name__)
from gui.control_panel import ControlPanel
from gui.dashboard import ProgressPanel, SpiralPlotPanel, MassExtractionPanel
from gui.results_panel import build_results_dict, get_status


class MainWindow:
    """
    Main application window for the Higgs propagator calculation GUI.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Higgs Propagator Calculation - Reuter Spiral Background")
        self.root.geometry("1400x900")

        # State
        self._integration_result = None
        self._pole_result = None
        self._running = False

        # Ensure output dirs exist and attach a file log handler for the GUI
        setup_output_dirs()
        self._setup_file_logging()

        self._build_ui()

    def _setup_file_logging(self):
        """Attach a FileHandler to the root logger writing to output/gui.log."""
        log_path = output_path("gui.log")
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logging.getLogger().addHandler(fh)
        logger.info("GUI session started — log: %s", log_path)

    def _build_ui(self):
        # Main layout: sidebar + main content
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left sidebar
        sidebar_frame = ttk.Frame(main_pane, width=220)
        main_pane.add(sidebar_frame, weight=0)

        # Control panel in sidebar
        self.control_panel = ControlPanel(
            sidebar_frame,
            on_run=self._on_run,
            on_scan=self._on_scan,
            on_convergence=self._on_convergence,
            on_reset=self._on_reset,
            on_export=self._on_export,
        )
        self.control_panel.pack(fill=tk.BOTH, expand=True)

        # Right content area
        content_frame = ttk.Frame(main_pane)
        main_pane.add(content_frame, weight=1)

        # Top: progress panel
        self.progress_panel = ProgressPanel(content_frame)
        self.progress_panel.pack(fill=tk.X, padx=4, pady=2)

        ttk.Separator(content_frame, orient="horizontal").pack(fill=tk.X, padx=4)

        # Notebook with tabs
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Tab 1: Dashboard
        dash_tab = ttk.Frame(self.notebook)
        self.notebook.add(dash_tab, text="Dashboard")
        self.spiral_panel = SpiralPlotPanel(dash_tab)
        self.spiral_panel.pack(fill=tk.BOTH, expand=True)
        self.mass_panel = MassExtractionPanel(dash_tab)
        self.mass_panel.pack(fill=tk.BOTH, expand=True)

        # Tab 2: Results
        results_tab = ttk.Frame(self.notebook)
        self.notebook.add(results_tab, text="Results")
        self._build_results_tab(results_tab)

        # Tab 3: Diagnostics log
        diag_tab = ttk.Frame(self.notebook)
        self.notebook.add(diag_tab, text="Diagnostics Log")
        self._build_diagnostics_tab(diag_tab)

    def _build_results_tab(self, parent):
        """Build the results summary tab."""
        ttk.Label(parent, text="HIGGS MASS EXTRACTION RESULTS",
                  font=("Courier", 11, "bold")).pack(pady=4)

        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.results_text = tk.Text(
            frame, font=("Courier", 9), state=tk.DISABLED,
            yscrollcommand=scrollbar.set, wrap=tk.NONE,
            bg="#1a1a2e", fg="#00ff88", height=30
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.results_text.yview)

        self._update_results_text(["Click 'RUN INTEGRATION' to start calculation."])

    def _build_diagnostics_tab(self, parent):
        """Build the diagnostics log tab."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.diag_text = tk.Text(
            frame, font=("Courier", 8), state=tk.DISABLED,
            yscrollcommand=scrollbar.set, wrap=tk.NONE,
            bg="#0a0a1a", fg="#88ff88", height=35
        )
        self.diag_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.diag_text.yview)

    def _update_results_text(self, lines):
        """Update the results text widget."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, "\n".join(lines))
        self.results_text.config(state=tk.DISABLED)

    def _log_diagnostic(self, message):
        """Append a message to the diagnostics log and write it to the log file."""
        logger.info(message)
        self.diag_text.config(state=tk.NORMAL)
        self.diag_text.insert(tk.END, message + "\n")
        self.diag_text.see(tk.END)
        self.diag_text.config(state=tk.DISABLED)

    # -------------------------------------------------------------------------
    # Action handlers
    # -------------------------------------------------------------------------

    def _on_run(self, params):
        if self._running:
            messagebox.showwarning("Busy", "Integration already running.")
            return
        self._running = True
        self.progress_panel.update(0, params.get("N_max", DEFAULT_N_MAX), status="Running...")
        self._log_diagnostic(f"Starting integration: theta_i={params['theta_i']:.4f}, "
                              f"theta_r={params['theta_r']:.4f}, N_max={params['N_max']:.4f}")

        def _run():
            try:
                result = run_integration(
                    theta_r=params["theta_r"],
                    theta_i=params["theta_i"],
                    N_max=params["N_max"],
                    rtol=params["rtol"],
                    atol=params["atol"],
                    termination=params["termination"],
                )
                self.root.after(0, lambda: self._on_integration_done(result, params))
            except Exception as e:
                self.root.after(0, lambda: self._on_integration_error(str(e)))

        threading.Thread(target=_run, daemon=True).start()

    def _on_integration_done(self, result, params):
        self._running = False
        self._integration_result = result

        if not result.get("success"):
            self._log_diagnostic(f"Integration FAILED: {result.get('message', 'Unknown error')}")
            messagebox.showerror("Integration Failed", result.get("message", "Unknown error"))
            self.progress_panel.update(0, params.get("N_max", DEFAULT_N_MAX), status="Failed")
            return

        stats = result.get("statistics", {})
        N_max = stats.get("N_max", DEFAULT_N_MAX)
        self.progress_panel.update(N_max, N_max, status="Extracting pole...")
        self._log_diagnostic(
            f"Integration complete: {stats.get('n_steps', 0):,} steps, "
            f"{stats.get('cpu_time', 0):.1f}s"
        )

        # Update trajectory plot
        t_arr = result["t"]
        G_arr = result["y"][0]
        Lambda_arr = result["y"][1]
        self.spiral_panel.update_plots(t_arr, G_arr, Lambda_arr)
        self.mass_panel.update_vev(t_arr)

        # Extract pole
        pole_result = extract_pole(result)
        self._pole_result = pole_result

        if pole_result.get("success"):
            m_h = pole_result["m_h"]
            self._log_diagnostic(f"Pole extracted: m_h = {m_h:.4f} GeV")
            self.mass_panel.update_propagator(
                pole_result["p2_arr"], pole_result["inv_prop_arr"],
                p2_pole=pole_result.get("p2_pole")
            )
        else:
            self._log_diagnostic("Pole extraction failed - no zero in search range")

        # Build and display results
        results_dict = build_results_dict(result, pole_result)
        lines = format_results_summary(results_dict)
        self._update_results_text(lines)

        self.progress_panel.update(N_max, N_max, status="Done")
        self.notebook.select(1)   # Switch to Results tab

    def _on_integration_error(self, error_msg):
        self._running = False
        self._log_diagnostic(f"ERROR: {error_msg}")
        messagebox.showerror("Error", error_msg)
        self.progress_panel.update(0, DEFAULT_N_MAX, status="Error")

    def _on_scan(self):
        self._log_diagnostic("Starting parameter scan (background)...")
        messagebox.showinfo(
            "Parameter Scan",
            "Parameter scan will run in background.\n"
            "Check Diagnostics Log for progress.\n"
            f"Results will be saved to output/parameter_scan_results.json"
        )

        def _run_scan():
            try:
                result = run_parameter_scan(n_jobs=2)
                self.root.after(0, lambda: self._on_scan_done(result))
            except Exception as e:
                self.root.after(0, lambda: self._log_diagnostic(f"Scan error: {e}"))

        threading.Thread(target=_run_scan, daemon=True).start()

    def _on_scan_done(self, result):
        sweet = result.get("sweet_spot", {})
        self._log_diagnostic(
            f"Scan complete. Sweet spot: theta_i={sweet.get('theta_i', '?'):.4f}, "
            f"N_max={sweet.get('N_max', '?'):.4f}, m_h={sweet.get('m_h', '?'):.3f} GeV"
        )
        json_path = export_results_json(result, "parameter_scan_results.json")
        self._log_diagnostic(f"Saved to {json_path}")

    def _on_convergence(self):
        params = self.control_panel.get_params()
        self._log_diagnostic("Starting convergence test...")

        def _run():
            try:
                result = run_convergence_test(
                    theta_r=params["theta_r"],
                    theta_i=params["theta_i"],
                    N_max=params["N_max"],
                )
                self.root.after(0, lambda: self._on_convergence_done(result))
            except Exception as e:
                self.root.after(0, lambda: self._log_diagnostic(f"Convergence error: {e}"))

        threading.Thread(target=_run, daemon=True).start()

    def _on_convergence_done(self, result):
        status = result.get("status", "UNKNOWN")
        mass = result.get("final_mass", float("nan"))
        unc = result.get("uncertainty", float("nan"))
        self._log_diagnostic(
            f"Convergence test: {status} | m_h = {mass:.4f} +/- {unc:.4f} GeV"
        )
        for d in result.get("details", []):
            self._log_diagnostic(
                f"  rtol={d['rtol']:.1e} atol={d['atol']:.1e}: "
                f"m_h={d['m_h']:.4f} GeV in {d['cpu_time']:.1f}s"
            )

    def _on_reset(self):
        self._log_diagnostic("Parameters reset to defaults.")

    def _on_export(self):
        if self._integration_result is None:
            messagebox.showwarning("No Data", "Run integration first.")
            return
        from utils.export import OUTPUT_DIR
        filepath = filedialog.asksaveasfilename(
            initialdir=OUTPUT_DIR,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Export Results"
        )
        if filepath:
            try:
                from gui.results_panel import build_results_dict
                results_dict = build_results_dict(self._integration_result, self._pole_result)
                export_results_json(results_dict, filepath)
                self._log_diagnostic(f"Results exported to {filepath}")

                # Also save dashboard PNG to output/plots/
                try:
                    from gui.plots import create_full_dashboard
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    fig = create_full_dashboard(
                        self._integration_result, self._pole_result
                    )
                    png_path = save_figure_png(fig, "dashboard.png")
                    plt.close(fig)
                    self._log_diagnostic(f"Dashboard PNG saved to {png_path}")
                except Exception as plot_exc:
                    self._log_diagnostic(f"Could not save dashboard PNG: {plot_exc}")

                messagebox.showinfo("Export", f"Data saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Export Error", str(e))


def launch_gui():
    """Launch the main GUI application."""
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()
