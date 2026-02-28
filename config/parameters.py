"""Default parameter values and physical constants for FRGE Higgs propagator calculation."""

import numpy as np

# =============================================================================
# Critical exponents (Falls-Litim 2014)
# =============================================================================
THETA_R = 2.714   # Real part of critical exponent
THETA_I = 2.396   # Imaginary part of critical exponent

# =============================================================================
# Fixed point coordinates (Reuter NGFP)
# =============================================================================
G_STAR = 0.707        # Dimensionless Newton coupling at fixed point
LAMBDA_STAR = 0.193   # Dimensionless cosmological constant at fixed point

# =============================================================================
# Initial conditions for spiral trajectory
# =============================================================================
A_G = 0.05         # Initial amplitude for G perturbation
A_LAMBDA = 0.03    # Initial amplitude for Lambda perturbation
DELTA_G = 0.0      # Initial phase for G spiral
DELTA_LAMBDA = np.pi / 4.0  # Initial phase for Lambda spiral

# =============================================================================
# Physical scales (in GeV)
# =============================================================================
M_PLANCK = 1.22e19    # Planck mass in GeV
HIGGS_VEV = 246.22    # Higgs vacuum expectation value in GeV
HIGGS_MASS_EXP = 125.25   # Experimental Higgs mass in GeV (PDG 2023)
HIGGS_MASS_ERROR = 0.17   # Experimental uncertainty in GeV

# =============================================================================
# Numerical integration parameters
# =============================================================================
DEFAULT_RTOL = 1e-8
DEFAULT_ATOL = 1e-10
DEFAULT_N_MAX = np.log(M_PLANCK / HIGGS_VEV)   # ~39.34 e-folds
RESONANCE_N_MAX = 30.0 * np.pi / THETA_I        # ~39.336 e-folds

# Spiral period and step control
T_RG = 2.0 * np.pi / THETA_I                   # ~2.622 e-folds
DEFAULT_MAX_STEP = T_RG / 100.0                 # ~0.026 e-folds

# =============================================================================
# Target predictions
# =============================================================================
HIGGS_MASS_THEORY = (4.0 / 3.0) * (THETA_I / (2.0 * np.pi)) * HIGGS_VEV  # ~125.19 GeV
ENHANCEMENT_FACTOR = 4.0 / 3.0
EXPECTED_MIXING_RATIO = 1.0 / 3.0
EXPECTED_SPHERE_RATIO = 4.0 / 3.0

# =============================================================================
# Matter field initial conditions
# =============================================================================
H0 = 1.0       # Initial radial Higgs field (dimensionless, in units of v)
OMEGA0 = 0.0   # Initial Goldstone angle

# Higgs self-coupling parameters
LAMBDA_H_INIT = 0.0     # Higgs quartic at Planck scale (Shaposhnikov-Wetterich scenario)
MASS_PARAM_INIT = 0.0   # Higgs mass parameter at Planck scale

# =============================================================================
# Pole search parameters
# =============================================================================
POLE_SEARCH_MIN = -(130.0 ** 2)   # p^2 lower bound (GeV^2)
POLE_SEARCH_MAX = -(120.0 ** 2)   # p^2 upper bound (GeV^2)
POLE_SEARCH_POINTS = 200          # Number of evaluation points

# =============================================================================
# Success criteria  (Part 7 of specification)
# =============================================================================
HIGGS_MASS_TARGET_MIN  = 125.1   # GeV  ─ lower bound of CONFIRMED window (spec)
HIGGS_MASS_TARGET_MAX  = 125.3   # GeV  ─ upper bound of CONFIRMED window (spec)
HIGGS_MASS_MARGINAL_MIN = 124.0  # GeV  ─ lower bound of MARGINAL window
HIGGS_MASS_MARGINAL_MAX = 126.5  # GeV  ─ upper bound of MARGINAL window

# =============================================================================
# Diagnostic thresholds
# =============================================================================
PHASE_SENSITIVITY_THRESHOLD = 0.1    # GeV  ─ spec: std(m_h) < 0.1 GeV over period
CONVERGENCE_THRESHOLD = 0.005        # GeV  ─ spec: Δm_h < 0.01 GeV (0.005 is stricter)
MIXING_RATIO_TARGET   = 1.0 / 3.0   # spec: ≈ 1/3
MIXING_RATIO_TOLERANCE = 0.05        # spec: ± 0.05
MIXING_RATIO_MIN = MIXING_RATIO_TARGET - MIXING_RATIO_TOLERANCE   # ≈ 0.283
MIXING_RATIO_MAX = MIXING_RATIO_TARGET + MIXING_RATIO_TOLERANCE   # ≈ 0.383
SPHERE_RATIO_TARGET   = 4.0 / 3.0   # spec: ≈ 4/3
SPHERE_RATIO_TOLERANCE = 0.05        # spec: ± 0.05
SPHERE_RATIO_MIN = SPHERE_RATIO_TARGET - SPHERE_RATIO_TOLERANCE   # ≈ 1.283
SPHERE_RATIO_MAX = SPHERE_RATIO_TARGET + SPHERE_RATIO_TOLERANCE   # ≈ 1.383

# =============================================================================
# Parameter scan ranges  (used by run_parameter_scan and run_full_grid_scan)
# =============================================================================

# ── existing 2-D grid ─────────────────────────────────────────────────────────
SCAN_THETA_I_MIN = 2.30
SCAN_THETA_I_MAX = 2.50
SCAN_THETA_I_POINTS = 20

SCAN_N_MAX_MIN = 38.0
SCAN_N_MAX_MAX = 40.0
SCAN_N_MAX_POINTS = 20

# ── additional axes for full 3-D grid ─────────────────────────────────────────
SCAN_THETA_R_MIN    = 2.50
SCAN_THETA_R_MAX    = 3.00
SCAN_THETA_R_POINTS = 15

# ── 1-D sensitivity sweep ranges ──────────────────────────────────────────────
# A_G  (initial Newton-coupling perturbation amplitude)
SCAN_A_G_MIN    = 0.01
SCAN_A_G_MAX    = 0.15
SCAN_A_G_POINTS = 20

# A_Lambda  (initial cosmological-constant perturbation amplitude)
SCAN_A_LAMBDA_MIN    = 0.005
SCAN_A_LAMBDA_MAX    = 0.10
SCAN_A_LAMBDA_POINTS = 20

# delta_Lambda  (initial phase of Lambda spiral component)
SCAN_DELTA_LAMBDA_MIN    = 0.0
SCAN_DELTA_LAMBDA_MAX    = np.pi / 2.0
SCAN_DELTA_LAMBDA_POINTS = 20

# ── Monte Carlo uncertainty propagation ───────────────────────────────────────
MC_THETA_I_SIGMA = 0.05
MC_N_SAMPLES = 200
