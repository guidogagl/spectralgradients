import torch
import torch.nn as nn

from tqdm.autonotebook import tqdm


class Complexity(nn.Module):
    def __init__(
        self,
        f: nn.Module = None,
        exp: nn.Module = None,
        name: str = "comp",
        **kwargs,
    ):
        super(Complexity, self).__init__()
        self.name = name

    @torch.no_grad()
    def forward(self, x, attr, mask):
        C_i = torch.abs(attr) + 1e-8
        # C_i batch_size , m , n
        C = C_i.sum(dim=-1)  # batch_size, m
        C_i = torch.einsum("bmn,bm->bmn", C_i, 1 / C)
        C = -torch.einsum("bmn,bmn->bm", C_i, torch.log(C_i))
        return C



class TFComplexity(nn.Module):
    def __init__(
        self,
        f: nn.Module = None,
        exp: nn.Module = None,
        name: str = "tfcomp",
        **kwargs,
    ):
        super(TFComplexity, self).__init__()
        self.name = name

    @torch.no_grad()
    def forward(self, x, attr, mask):
        
        batch_size, m, _, _ = attr.shape
        attr = attr.reshape( batch_size, m, -1 )         
        
        C_i = torch.abs(attr) + 1e-8
        # C_i batch_size , m , n
        C = C_i.sum(dim=-1)  # batch_size, m
        C_i = torch.einsum("bmn,bm->bmn", C_i, 1 / C)
        C = -torch.einsum("bmn,bmn->bm", C_i, torch.log(C_i))
        return C
