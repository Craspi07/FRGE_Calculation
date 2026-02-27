"""Diagnostics package for FRGE Higgs propagator calculation."""
from .convergence import run_convergence_test
from .mixing_analysis import analyze_mixing_term
from .sphere_integral import verify_sphere_enhancement
from .phase_sensitivity import check_phase_sensitivity
from .parameter_scan import run_parameter_scan
