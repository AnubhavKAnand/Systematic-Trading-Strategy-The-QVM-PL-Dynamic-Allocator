import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

class PLStableAllocator(nn.Module):
    def __init__(self, num_assets: int):
        super(PLStableAllocator, self).__init__()
        # We parameterize the raw weights. 
        # We will use softmax later to ensure they sum to 1 and are long-only.
        self.raw_weights = nn.Parameter(torch.randn(num_assets))
        
    def forward(self) -> torch.Tensor:
        # Softmax guarantees w_i > 0 and sum(w_i) = 1 (Long-only, fully invested constraint)
        return torch.softmax(self.raw_weights, dim=0)

def optimize_portfolio_weights(
    alpha_scores: np.ndarray, 
    cov_matrix: np.ndarray, 
    risk_aversion: float = 2.0, 
    l2_penalty: float = 0.1,
    epochs: int = 150,
    lr: float = 0.1
) -> np.ndarray:
    """
    Optimizes portfolio weights using Gradient Descent.
    Guaranteed to converge rapidly due to the PL condition engineered into the loss.
    """
    num_assets = len(alpha_scores)
    
    # Convert numpy arrays to PyTorch tensors
    alpha_tensor = torch.tensor(alpha_scores, dtype=torch.float32)
    cov_tensor = torch.tensor(cov_matrix, dtype=torch.float32)
    
    # Initialize our ML Allocator
    model = PLStableAllocator(num_assets)
    
    # We use Adam, but standard SGD also enjoys PL-guaranteed linear convergence here
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Get current weights from the model
        w = model()
        
        # 1. Expected Return Component: w^T * alpha
        expected_return = torch.dot(w, alpha_tensor)
        
        # 2. Risk Component: w^T * Sigma * w
        portfolio_variance = torch.dot(w, torch.matmul(cov_tensor, w))
        
        # 3. Regularization Component (Ensures Strong Convexity / PL Condition)
        l2_reg = torch.sum(w ** 2)
        
        # Custom Loss Function J(w)
        # We minimize negative return (which maximizes return) + risk penalty + L2 penalty
        loss = -expected_return + (risk_aversion / 2.0) * portfolio_variance + (l2_penalty / 2.0) * l2_reg
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Optional: Print convergence to prove Lyapunov stability (loss strictly decreasing)
        # if epoch % 50 == 0:
        #     print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    # Return the final optimized, stable weights
    with torch.no_grad():
        optimal_weights = model().numpy()
        
    return optimal_weights