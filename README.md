# Higgs Propagator Calculation in Reuter Spiral Background

Production-ready implementation of the functional renormalization group (FRGE)
propagator calculation for the Higgs boson mass in the Reuter non-Gaussian fixed
point (NGFP) spiral background.

## Theory Overview

The Reuter NGFP in asymptotically safe quantum gravity has complex critical exponents:

    θ = θ_r ± i·θ_i = 2.714 ± 2.396·i

This creates logarithmic spiral trajectories in coupling space as the RG scale k
runs from the Planck scale M_Pl down to the electroweak scale v.

**Target Prediction:**

    m_h = (4/3) × (θ_i/2π) × v = 125.19 GeV

where the 4/3 factor arises from the composite manifold M = [0,A₀] × S²
(radial Higgs direction × Goldstone sphere).

**Experimental Value:** 125.25 ± 0.17 GeV (PDG 2023)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd FRGE_Calculation

# Install Python 3.10+, then:
pip install -r requirements.txt

# Launch the GUI
python main.py
```

## Usage

### GUI Mode (default)
```bash
python main.py
```
Opens the full dashboard with real-time monitoring, parameter sliders,
and result visualization.

### Headless Mode
```bash
python main.py --headless
python main.py --headless --theta-i 2.396 --theta-r 2.714 --N-max 39.34
```

### Run Tests
```bash
python main.py --test
# or directly:
pytest tests/ -v
```

### Performance Benchmarks
```bash
python main.py --benchmark
```

### Parameter Scan
```bash
python main.py --scan
```
Runs a 2D scan over (θ_i, N_max) space and saves to `parameter_scan_results.json`.

## Package Structure

```
FRGE_Calculation/
├── main.py                 # Entry point and CLI
├── requirements.txt
├── README.md
├── config/
│   ├── parameters.py       # Physical constants and default values
│   └── presets.py          # Named parameter presets
├── core/
│   ├── wetterich.py        # Wetterich FRGE implementation
│   ├── beta_functions.py   # RG beta functions (gravity + matter)
│   ├── spiral.py           # Spiral trajectory computation
│   ├── york.py             # York decomposition & composite manifold
│   ├── integrator.py       # DOP853 adaptive RG flow integrator
│   ├── pole_finder.py      # Brent + cubic spline pole extraction
│   └── hessian.py          # Second variation (Hessian) computations
├── gui/
│   ├── main_window.py      # Main tkinter window
│   ├── control_panel.py    # Parameter sliders and buttons
│   ├── dashboard.py        # Real-time monitoring plots
│   ├── plots.py            # Matplotlib plot utilities
│   └── results_panel.py    # Formatted results display
├── diagnostics/
│   ├── convergence.py      # Multi-level tolerance convergence testing
│   ├── mixing_analysis.py  # Off-diagonal Hessian diagnostics
│   ├── sphere_integral.py  # Monte Carlo S² enhancement verification
│   ├── phase_sensitivity.py # Phase dependence analysis
│   └── parameter_scan.py   # 2D parameter space exploration
├── utils/
│   ├── constants.py        # Physical constants
│   ├── math_utils.py       # Numerical utilities
│   ├── export.py           # JSON/NPZ data export
│   └── validation.py       # Parameter and result validation
└── tests/
    ├── test_wetterich.py
    ├── test_integrator.py
    ├── test_pole_finder.py
    └── benchmarks.py
```

## Key Physics

### Wetterich Equation
```
∂_t Γ_k = (1/2) STr[(Γ_k^(2) + R_k)^{-1} · ∂_t R_k]
```
with the Litim optimized regulator R_k(p²) = (k² - p²)·θ(k² - p²).

### Spiral Trajectory
```
G̃(t) = G̃* + A_G · exp(-θ_r·t) · cos(θ_i·t + δ_G)
Λ̃(t) = Λ̃* + A_Λ · exp(-θ_r·t) · sin(θ_i·t + δ_Λ)
```
Fixed point: (G̃*, Λ̃*) ≈ (0.707, 0.193)

### 4/3 Enhancement
The composite manifold M = [0,A₀] × S² yields:
```
m_eff² = Γ_hh · (1 + Γ_ww/Γ_hh)  ≈  Γ_hh · (4/3)
```
The off-diagonal Hessian elements ∂²Γ/∂h∂ω encode the Higgs-Goldstone mixing.

## Integration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| θ_i | 2.396 | Imaginary critical exponent |
| θ_r | 2.714 | Real critical exponent |
| N_max | 39.34 | Total RG time (e-folds) |
| rtol | 1e-8 | Relative ODE tolerance |
| atol | 1e-10 | Absolute ODE tolerance |
| Method | DOP853 | 8th-order Runge-Kutta |

## Success Criteria

- 125.0 ≤ m_h ≤ 125.4 GeV
- |m_h - 125.19| ≤ 0.2 GeV (theory agreement)
- |m_h - 125.25| ≤ 0.17 GeV (experimental agreement)
- Off-diagonal mixing ratio: 0.28 ≤ ratio ≤ 0.38 (expected: 1/3)
- S² enhancement factor: 1.28 ≤ factor ≤ 1.38 (expected: 4/3)
- Phase sensitivity: std(m_h) < 0.1 GeV over complete period

## Interpreting Results

- **[+ CONFIRMED]** Green status: Theory validated within precision
- **[~ MARGINAL]** Yellow status: Close but needs investigation
- **[X FAILED]** Red status: Significant discrepancy or numerical error

## Troubleshooting

- **Integration hangs**: Reduce `N_max` or use `fast_preview` preset
- **Mass far from target**: Check parameters are near defaults; try `resonance` termination
- **GUI slow**: matplotlib rendering; reduce window size or use `--headless`
- **No pole found**: Widen the search range in `config/parameters.py` (POLE_SEARCH_MIN/MAX)

## References

1. Reuter, M. (1998). *Nonperturbative evolution equation for quantum gravity.*
   Phys. Rev. D 57, 971.
2. Falls, K. & Litim, D. (2014). *Black hole thermodynamics under the microscope.*
   Phys. Rev. D 89, 084046. (Critical exponents: θ = 2.714 ± 2.396i)
3. Shaposhnikov, M. & Wetterich, C. (2010). *Asymptotic safety of gravity and the Higgs.*
   Phys. Lett. B 683, 196. (m_h ≈ 126 GeV from λ(M_Pl) = 0)
4. Wetterich, C. (1993). *Exact evolution equation for the effective potential.*
   Phys. Lett. B 301, 90.
