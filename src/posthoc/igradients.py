import torch
import torch.nn as nn

from physioex.explain.posthoc._trapezoid import segment_fun, trapezoid_rule

from typing import Union

class IntegratedGradients(nn.Module):
    def __init__(self, 
        f : callable,
        baseline : Union[float, torch.Tensor] = float(0),
        n_points : int = 50,
        ):
        super().__init__()

        # if baseline is a torch tensor register it as a buffer
        if isinstance(baseline, torch.Tensor):
            self.register_buffer('baseline', baseline)
        elif isinstance(baseline, float) or baseline is None:
            self.baseline = baseline
        else:
            raise ValueError("Baseline must be a float or a torch tensor")
        
        self.fun = f
        self.n_points = n_points
        
    def forward(self, x, baseline : Union[float, torch.Tensor] = None):
        
        batch_size, n = x.shape
        
        if self.baseline is None:
            assert baseline is not None, "Baseline must be provided if not set during initialization"    

        if baseline is None:
            baseline = self.baseline
        
        # if baseline is a float then it is a constant baseline
        if isinstance(baseline, float):
            baseline = torch.ones_like(x, device=x.device) * baseline
            
        elif isinstance(baseline, torch.Tensor):
            if len( baseline.shape ) == 1:
                # baseline is of dimension n, we need to repeat it for the batch size
                assert baseline.shape[0] == n, "Baseline must have the same dimension as the input N"
                baseline = baseline.unsqueeze(0).expand(batch_size, n)
            elif len( baseline.shape ) == 2:
                assert baseline.shape == x.shape, "Baseline must have the same dimension as the input [B, N]"    
            
        return trapezoid_rule( self.fun, x, baseline, segment_fun, self.n_points )