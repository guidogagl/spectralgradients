import torch
import torch.nn as nn

from tqdm.autonotebook import tqdm


class IROF(nn.Module):
    def __init__(
        self,
        f: callable,
        patch: float = 0,  # value to substitute the feature with
        n_features: int = 1,  # number of feature to remove at each step
    ):
        super(IROF, self).__init__()
        self.f = f
        self.patch = patch
        self.n_features = n_features

    @torch.no_grad()
    def forward(self, x, attr, mask):

        sorted_attr = torch.argsort(attr, dim=-1)

        irof = [ self.f(x).cpu() ]

        for step, i in enumerate( range(1, x.shape[-1], self.n_features)):

            if i + self.n_features > x.shape[-1]:
                n_features = x.shape[-1] - i
            else:
                n_features = self.n_features

            for b in range(0, x.shape[0]):
                x[b, sorted_attr[b, :,  i : i + n_features].long()] = self.patch

            irof = irof +  [self.f(x).cpu()]
        
        irof = torch.stack( irof, dim = 0 ).to(x.device)

        irof = torch.trapezoid( irof, dx = 1/irof.shape[0], dim = 0)

        return irof