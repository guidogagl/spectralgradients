import torch
import torch.nn as nn

from src.posthoc.integrals import step_integral

from typing import Union

def expected_line_integral(
    fn: nn.Module,
    x : torch.Tensor,
    baselines : torch.Tensor
):
    n_points = len(baselines)

    SG = 0
    t = torch.linspace(
        0, 1, n_points, device=x.device, dtype=x.dtype
    )

    for i in range(1, n_points):
        
        def path( alpha ):
            return x*alpha - ( 1-alpha) * baselines[i].to(x.device)
        
        def jac_path( alpha ):
            return x - baselines[i].to(x.device)

        SG += step_integral(fn, t, i, path, jac_path, x.device)
            
    return SG

def expected_Lineh_integral(
    fn: nn.Module,
    x : torch.Tensor,
    baselines : torch.Tensor,
):

    T = torch.linspace(
        0, 1, baselines.shape[0], device=x.device, dtype=x.dtype
    )

    SG = 0
    
    curve = []

    for i, t in enumerate(T):
        
        path_step = baselines[i] - t * ( baselines[i] - x ) 

        G = torch.func.vmap( torch.func.jacrev(fn))( path_step ) 

        J = - ( baselines[i] - x ) 

        G = torch.einsum( "bmn,bn->bmn", G, J )

        curve = curve + [G]        

    curve = torch.stack( curve, dim = 0)    
    curve = torch.trapezoid( curve, dx = 1/(baselines.shape[0]-1), dim = 0 )

    return curve

class ExpectedGradients(nn.Module):
    def __init__(
        self,
        fn : nn.Module,
        baselines : torch.Tensor,
        n_points : int = 50,    
    ):
        super().__init__()

        self.baselines = []

        for i in range( n_points ):
            indx = torch.randperm( len(baselines))
            self.baselines += baselines[indx]
        
        self.baselines = torch.stack( self.baselines, dim = 0 ).reshape(n_points, len(baselines),  *baselines.shape[1:])
        
        self.n_points = n_points
        self.fn = fn

    def forward(self, x):
        batch_size = x.shape[0]
        return expected_line_integral(self.fn, x, self.baselines[:, :batch_size])
