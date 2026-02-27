"""Core physics modules for FRGE Higgs propagator calculation."""
from .integrator import run_integration
from .pole_finder import extract_pole
from .hessian import compute_hessian
