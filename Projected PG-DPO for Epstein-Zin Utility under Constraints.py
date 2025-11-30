# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 18:37:01 2025

@author: WONCHAN
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuration & Physics (EZ Utility Parameters)
class Config:
    def __init__(self):
        # Market Params
        self.T = 1.0                # Time horizon
        self.N_steps = 50           # Time discretization steps
        self.dt = self.T / self.N_steps
        self.dim = 2                # Number of assets
        self.r = 0.03               # Risk-free rate
        self.mu = torch.tensor([0.10, 0.12]) # Drift
        self.sigma = torch.tensor([[0.20, 0.0], [0.0, 0.30]]) # Volatility matrix
        
        # EZ Utility Params (Based on Tian et al.)
        self.beta = 0.05            # Time preference
        self.gamma = 4.0            # Risk aversion (R)
        self.psi = 0.5              # EIS (1/S, typically < 1 implies psi < 1)
        self.theta = (1 - self.gamma) / (1 - 1/self.psi)
        
        # Constraints
        self.leverage_limit = 1.5   # Maximum sum of weights
        self.transaction_cost = 0.0 # Future work extension

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 64

# 2. Mathematical Core: Aggregator & Projection
class MathCore:
    def __init__(self, cfg):
        self.cfg = cfg

    def ez_aggregator(self, c, v):
        """
        Calculates the driver -f(c, v).
        Using the standard Duffie-Epstein-Zin aggregator form.
        f(c, v) = beta * theta * v * [ (c/v)^(1/psi) - 1 ]
        """
        # Simplified form for demonstration (verify with paper's exact eq):
        # f(c, V) = \frac{\beta}{1 - 1/\psi} (1-\gamma)V [ ( \frac{c}{ ((1-\gamma)V)^{\frac{1}{1-\gamma}} } )^{1 - 1/\psi} - 1 ]
                
        term1 = (c / ((1 - self.cfg.gamma) * v).pow(1/(1-self.cfg.gamma)) ).pow(1 - 1/self.cfg.psi)
        driver = (self.cfg.beta / (1 - 1/self.cfg.psi)) * (1 - self.cfg.gamma) * v * (term1 - 1)
        return driver

    def project_strategy(self, z_t, v_t, w_t):
        """
        THE KEY CONTRIBUTION: Projected PG-DPO
        Instead of NN outputting pi, NN outputs Z (volatility of utility).
        We calculate pi* using FOC and then PROJECT it onto constraints.
        """
        # 1) Theoretical Unconstrained Pi (Merton-EZ hybrid)
        # pi* = (1/gamma) * (sigma*sigma.T)^-1 * (mu - r) + Hedging_Term(z_t)
        
        sigma_inv = torch.inverse(self.cfg.sigma)
        sigma_sq_inv = torch.inverse(self.cfg.sigma @ self.cfg.sigma.t())
        
        myopic_demand = (1 / self.cfg.gamma) * (sigma_sq_inv @ (self.cfg.mu - self.cfg.r))
        
        # Hedging demand comes from Z (volatility of V)
        # pi_hedge = (1 / gamma) * (sigma.T)^-1 * (Z_t / V_t) (approx)
        # Ensure shapes match for batch processing
        hedging_demand = (1 / self.cfg.gamma) * (sigma_inv.t() @ z_t.t()).t() / v_t
        
        pi_unc = myopic_demand + hedging_demand 

        # 2) Projection (Constraint Handling)
        leverage = torch.sum(torch.abs(pi_unc), dim=1, keepdim=True)
        scale_factor = torch.clamp(self.cfg.leverage_limit / (leverage + 1e-8), max=1.0)
        
        pi_constrained = pi_unc * scale_factor
        
        return pi_constrained

    def optimal_consumption(self, v_t):
        # c* is usually a fraction of wealth or related to V_t in EZ
        # For Skeleton, assume simple optimal consumption rule derived from FOC
        # c* = ((1-gamma)V)^(1/(1-gamma)) * beta^psi ...
        # Simplified:
        c_star = 0.05 * v_t 
        return c_star

# 3. Neural Network (Approximator)
class BSDE_Net(nn.Module):
    def __init__(self, dim, output_dim):
        super(BSDE_Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 64), # Input: t, W_t (or state X_t)
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim) # Output: Z_t (vector)
        )
        
    def forward(self, t, x):
        # x is Wealth or State
        inputs = torch.cat([t, x], dim=1)
        return self.net(inputs)

# 4. Main Algorithm: Projected PG-DPO Solver
class ProjectedPGDPO:
    def __init__(self, cfg):
        self.cfg = cfg
        self.math = MathCore(cfg)
        
        # Learnable Initial Utility V_0
        self.V0 = nn.Parameter(torch.tensor([1.0], device=cfg.device))
        
        # Policy/Z Network
        self.z_net = BSDE_Net(dim=1, output_dim=cfg.dim).to(cfg.device)
        
        self.optimizer = optim.Adam(list(self.z_net.parameters()) + [self.V0], lr=0.005)

    def train_step(self):
        batch_size = self.cfg.batch_size
        t_stamp = torch.arange(0, self.cfg.T, self.cfg.dt, device=self.cfg.device)
        
        # Initial State
        W = torch.ones(batch_size, 1, device=self.cfg.device) # W0 = 1
        V = self.V0.expand(batch_size, 1) # Start with guessed V0
        
        dW = torch.randn(batch_size, len(t_stamp), self.cfg.dim, device=self.cfg.device) * np.sqrt(self.cfg.dt)
        
        for i in range(len(t_stamp)):
            t = t_stamp[i].reshape(-1, 1).expand(batch_size, 1)
            
            # (1) Neural Net Estimate of Z (Volatility of Utility)
            Z = self.z_net(t, W) # [Batch, Dim]
            
            # (2) Projection Step (Get Pi from Z)
            pi = self.math.project_strategy(Z, V, W) # [Batch, Dim]
            c = self.math.optimal_consumption(V)     # [Batch, 1]
            
            # (3) Forward Dynamics (Euler-Maruyama)
            
            # Wealth Process dW
            # dW = W * (r + pi(mu-r))dt - c dt + W pi sigma dZ
            mu_term = self.cfg.mu - self.cfg.r
            drift_W = (self.cfg.r * W + W * (pi @ mu_term) - c) * self.cfg.dt
            diff_W = W * (pi @ self.cfg.sigma) * dW[:, i, :] 
            diffusion_term = torch.sum(diff_W, dim=1, keepdim=True) 
            
            W_next = W + drift_W + diffusion_term
            
            # Utility Process dV (BSDE)
            # dV = -f(c, V)dt + Z dW_market
            driver = self.math.ez_aggregator(c, V)
            drift_V = -driver * self.cfg.dt
            diff_V = torch.sum(Z * dW[:, i, :], dim=1, keepdim=True)
            
            V_next = V + drift_V + diff_V
            
            # Update
            W = W_next
            V = V_next
            
        # 4) Loss Function
        # We want to find V0 and Z such that V_T = 0 (or Utility of terminal wealth)
        # In infinite horizon / consumption problem, usually V_T should converge or match boundary.
        # For finite horizon terminal wealth utility U(W_T) = V_T
        # Let's assume V_T should be 0 for pure consumption stream problem or U(W_T)
        
        target_V_T = 0.0 
        loss = torch.mean((V - target_V_T)**2)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), self.V0.item()

    def train(self, epochs=1000):
        for epoch in range(epochs):
            loss, v0_est = self.train_step()
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}, Est V0: {v0_est:.4f}")

# 5. Execution
if __name__ == "__main__":
    cfg = Config()
    solver = ProjectedPGDPO(cfg)
    print("Start Training Projected PG-DPO for EZ Utility...")
    solver.train(epochs=500)
    print("Training Complete.")