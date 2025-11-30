# Projected PG-DPO for Epstein-Zin Utility (WIP)

![Status](https://img.shields.io/badge/Status-Work_in_Progress-yellow)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Project Overview

This repository contains research code for **Projected PG-DPO (Pontryagin-Guided Direct Policy Optimization)** applied to **Epstein–Zin recursive utility** with constraints on consumption–investment decisions. The goal is to move beyond the classical CRRA/Merton setting by combining the **BSDE structure of EZ utility** with the **Pontryagin Maximum Principle (PMP)** to learn policies in high-dimensional, constrained asset environments.

> **WIP:** The codebase is under active research and development; algorithms and implementation details may change quickly.

## Key Contributions

- **Projected PG-DPO + EZ Utility:** Adapts PMP-driven PG-DPO to **Epstein–Zin utility**, respecting EZ-specific features such as the separation between risk aversion and intertemporal elasticity of substitution (IES).
- **Policy recovery under constraints:** Learn utility volatility \(Z_t\), then invert the **first-order conditions (FOC)** to obtain constrained investment and consumption policies.
- **Closed-form benchmark:** Use the unconstrained **closed-form solution** as both **(1) a projection guide** and **(2) a validation baseline** to stabilize training and verify accuracy.

## Methodology

1. **Mathematical framework**
   - Objective: maximize **Epstein–Zin recursive utility** \(V_0\) defined via a BSDE.
   - Constraints: leverage limit \(\|\pi\|_1 \le L\); transaction costs (planned extension).
   - Algorithm: **Projected PG-DPO** — estimate \(Z_t\) with a neural network, invert FOC to recover \(\pi^*\), and apply projection when constraints are present.

2. **Role of closed-form solutions**
   - **Projection anchor:** The unconstrained solution provides a base policy that guides early training and stabilizes exploration.
   - **Validation benchmark:** In the unconstrained setting, compare learned policies/values to the **Merton-style closed form** to check theoretical consistency.

## Status & Roadmap

- [x] **Theory:** Derive PMP/costate dynamics for EZ utility.
- [x] **Baseline:** Implement Deep BSDE training loop and PG-DPO skeleton.
- [x] **Unconstrained validation:** Verify agreement with the closed-form solution.
- [ ] **Constraint integration:** Implement L1 leverage constraint and projection layer (**current focus**).
- [ ] **Enhancements:** Add transaction costs, high-dimensional assets, and experiment automation.

## Installation

```bash
# Clone
git clone https://github.com/your-username/projected-pg-dpo-ez.git
cd projected-pg-dpo-ez

# Dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Unconstrained benchmark (closed-form validation)
python train.py --mode unconstrained --epochs 1000

# Leverage constraint applied
python train.py --mode constrained --leverage_limit 1.5
```

## Repo Structure (example)

- `train.py`: Training entry point (mode-specific configs).
- `models/`: BSDE-based value/volatility networks.
- `algos/`: PG-DPO core, projection operators, and FOC inversion logic.
- `utils/`: Data generation, logging, and seeding utilities.

## References

- Huh, J., et al., *"Breaking the Dimensional Barrier: A Pontryagin-Guided Direct Policy Optimization..."* (2025)
- Tian, D., et al., *"Optimal Consumption-Investment with Epstein-Zin Utility under Leverage Constraint"* (2025)
- Herdegen, M., et al., *"The infinite-horizon investment-consumption problem for Epstein-Zin stochastic differential utility"* (2022)

For questions or collaboration proposals, please open an Issue or Pull Request.
