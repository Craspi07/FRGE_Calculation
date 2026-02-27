"""Physical constants and unit conversions."""

import numpy as np

# Planck units
M_PLANCK_GEV = 1.22e19       # Planck mass in GeV
M_PLANCK_TEV = 1.22e16       # Planck mass in TeV
HBAR_C = 0.197326980e-15     # hbar*c in GeV*m

# Electroweak scale
HIGGS_VEV_GEV = 246.22       # Higgs VEV in GeV
M_W_GEV = 80.377             # W boson mass in GeV
M_Z_GEV = 91.1876            # Z boson mass in GeV
M_TOP_GEV = 172.69           # Top quark mass in GeV

# Higgs mass
HIGGS_MASS_EXP_GEV = 125.25  # Experimental Higgs mass (PDG 2023)
HIGGS_MASS_ERR_GEV = 0.17    # Uncertainty

# Fine structure constants
ALPHA_EM = 1.0 / 137.036     # Electromagnetic fine structure constant
ALPHA_S_MZ = 0.1179          # Strong coupling at M_Z
SIN2_THETA_W = 0.23122       # Weak mixing angle sin^2(theta_W)

# Gravitational
G_NEWTON_GEV = 6.708e-39     # Newton's constant in GeV^-2

# RG scales
N_MAX_STANDARD = np.log(M_PLANCK_GEV / HIGGS_VEV_GEV)  # ~39.34 e-folds

# Numerical constants
PI = np.pi
TWO_PI = 2.0 * np.pi
FOUR_PI = 4.0 * np.pi
FOUR_THIRDS = 4.0 / 3.0

# Reuter NGFP values
G_STAR_NGFP = 0.707
LAMBDA_STAR_NGFP = 0.193
THETA_R_NGFP = 2.714
THETA_I_NGFP = 2.396
T_RG_NGFP = TWO_PI / THETA_I_NGFP   # Spiral period

# Theory prediction
HIGGS_MASS_THEORY_GEV = FOUR_THIRDS * (THETA_I_NGFP / TWO_PI) * HIGGS_VEV_GEV
