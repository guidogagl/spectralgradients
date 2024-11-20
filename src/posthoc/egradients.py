import torch
import torch.nn as nn

from physioex.explain.posthoc._trapezoid import trapezoid_rule

from typing import Union


class ExpectedGradients(nn.Module):
    def __init__(self, 
        f : callable,
        baselines : torch.Tensor,
        n_points : int = 50,
        ):
        super().__init__()

        self.baselines = baselines

        def segment_fun( alpha : torch.Tensor, x_1: torch.Tensor, x_2: torch.Tensor):

            batch_size = x_1.shape[0]

            # draw a random baseline for the batch
            indexes = torch.randint(0, baselines.shape[0], (batch_size,))
            x_2 = baselines[indexes].to(x_1.device)    
            
            return alpha*x_1 + (1-alpha)*x_2

        self.segment_fun = segment_fun
        
        self.fun = f
        self.n_points = n_points
        
    def forward(self, x ):
        
        batch_size, n = x.shape
        indexes = torch.randint(0, x.shape[0], (batch_size,)).to(x.device)  
            
        return trapezoid_rule( self.fun, x, self.baselines[indexes], self.segment_fun, self.n_points )