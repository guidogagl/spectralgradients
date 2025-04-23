import torch
import torch.nn as nn

from tqdm.autonotebook import tqdm


class Infidelity(nn.Module):
    def __init__(
        self,
        f: callable,
        patch: float = 0,  # value to substitute the feature with
        percentage: float = 0.05,  # number of feature to remove at each step
        name: str = "inf",
        **kwargs,
    ):
        super(Infidelity, self).__init__()
        self.f = f
        self.patch = patch
        self.percentage = percentage
        self.name = name

    @torch.no_grad()
    def forward(self, x, attr, mask):
        batch_size, m, n = attr.size()

        attr = torch.einsum( "bmn,bn->bmn", attr, torch.sign(x))
        attr = nn.functional.relu(attr)

        patch_size = int( n * self.percentage )

        attr = attr.reshape( batch_size, m, -1, patch_size)
        attr = attr.sum( dim = -1)
        
        sorted_attr = torch.argsort(attr, dim=-1, descending = True)

        infidelity = []

        inf0 = self.f(x) # b, m
        for c in range( m ):
            x_ = x.clone()
            inf = [inf0[:, c]]

            for i in range( sorted_attr.size(-1) ):            
                for b in range( batch_size ):
                    f = int( sorted_attr[b, c, i] * patch_size )
                    x_[b, f:f+patch_size] = self.patch                   

                inf = inf + [self.f(x_)[:, c]]

            inf = inf + [self.f(torch.ones_like(x_) * self.patch)[:, c]]
            inf = torch.stack(inf, dim=0) # f, b 
            inf = torch.einsum( "fb,b->fb", inf, 1/inf[0])
            inf = torch.trapezoid( inf, dx = 1/inf.shape[0], dim = 0)

            infidelity = infidelity + [inf]

        infidelity = torch.stack(infidelity, dim=0) # m, b
        infidelity = infidelity.permute( 1, 0) 

        return infidelity




class TFInfidelity(nn.Module):
    def __init__(
        self,
        f: callable,
        patch: float = 0,  # value to substitute the feature with
        percentage: float = 0.05,  # number of feature to remove at each step
        name: str = "tfinf",
        **kwargs,
    ):
        super(TFInfidelity, self).__init__()
        self.f = f
        self.patch = patch
        self.percentage = percentage
        self.name = name

    @torch.no_grad()
    def forward(self, x, attr, mask):
        batch_size, m, f, n = attr.size()

        attr = torch.einsum( "bmfn,bn->bmfn", attr, torch.sign(x))
        attr = nn.functional.relu(attr).sum( -1 )

        patch_size = int( f * self.percentage )

        attr = attr.reshape( batch_size, m, -1, patch_size)
        attr = attr.sum( dim = -1)
        
        sorted_attr = torch.argsort(attr, dim=-1, descending = True)

        infidelity = []

        inf0 = self.f(x) # b, m
        for c in range( m ):
            x_ = x.clone()
            inf = [inf0[:, c]]

            for i in range( sorted_attr.size(-1) ):            
                for b in range( batch_size ):
                    f = int( sorted_attr[b, c, i] * patch_size )
                    x_[b, f:f+patch_size] = self.patch                   

                inf = inf + [self.f(x_)[:, c]]

            inf = inf + [self.f(torch.ones_like(x_) * self.patch)[:, c]]
            inf = torch.stack(inf, dim=0) # f, b 
            inf = torch.einsum( "fb,b->fb", inf, 1/inf[0])
            inf = torch.trapezoid( inf, dx = 1/inf.shape[0], dim = 0)

            infidelity = infidelity + [inf]

        infidelity = torch.stack(infidelity, dim=0) # m, b
        infidelity = infidelity.permute( 1, 0) 

        return infidelity