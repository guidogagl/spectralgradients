import torch
import torch.nn as nn


class Saliency( nn.Module ):
    def __init__( self, f : callable ):
        super().__init__()
        self.f = f
        
    def forward( self, x ):
        # the saliency is the gradient of the function with respect to the input
        x = torch.autograd.functional.jacobian( self.f, x, strategy="reverse-mode" )
        
        # x dim is batch_size, m , batch_size, n
        return x.sum( dim = 2 )
    

class InputXGradient( nn.Module ):
    def __init__( self, f : callable ):
        super().__init__()
        self.f = f
        
    def forward( self, x ):
        grads = torch.autograd.functional.jacobian( self.f, x, strategy="reverse-mode" )
        grads = grads.sum( dim = 2 )
        # grads shape is batch_size, m, n
        
        # multiply the gradients with the input        
        return  torch.einsum( 'bmn, bn -> bmn', grads, x )