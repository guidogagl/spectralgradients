import torch

def segment_fun( alpha : torch.Tensor, x_1: torch.Tensor, x_2: torch.Tensor):
    return alpha*x_1 + (1-alpha)*x_2

def trapezoid_rule(
    fn : torch.nn.Module, # the function whose gradients we want to integrate f : R^n -> Rˆm ( batched ) 
    x : torch.Tensor, # the batched input tensor to start the integration from
    x_ : torch.Tensor, # the initial value of the input tensor
    line_func : callable, # the function to compute the line integral between two points in the input space, by default is the segment line
    n_points : int = 2 # the number of points to sample between x and x_
    ): 

    # fn : R^n -> R^m
    # r(t) : R -> R^n  
    
    r = lambda t : line_func( t, x, x_ )
    
    J = torch.zeros( x.shape[0], fn(x[0].unsqueeze(0) ).shape[-1], x.shape[-1], device = x.device, dtype=x.dtype )
    
    # trapezoidal rule in n_points:
    t = torch.linspace(0, 1, n_points, device = x.device, dtype=x.dtype)
    
    for i in range(1, n_points):
        t0 = t[i-1]
        t1 = t[i]
        
        J_r0 = torch.autograd.functional.jacobian( r, t0, vectorize=True, strategy="forward-mode" ).squeeze(-1)
        J_r1 = torch.autograd.functional.jacobian( r, t1, vectorize=True, strategy="forward-mode" ).squeeze(-1)

        J_f_r0 = torch.autograd.functional.jacobian( fn, r(t0), strategy="reverse-mode" ) 
        J_f_r1 = torch.autograd.functional.jacobian( fn, r(t1), strategy="reverse-mode" ) 
    
        # J_f shape is batch_size, m, batch_size, n    
        J_f_r0, J_f_r1 = J_f_r0.sum( dim = 2), J_f_r1.sum( dim = 2)
    
        J_0 = J_f_r0 * J_r0.unsqueeze(1)
        J_1 = J_f_r1 * J_r1.unsqueeze(1)
    
        J += (t1 - t0) * ( J_0 + J_1 )/ 2
        
    return J