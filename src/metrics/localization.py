import torch
import torch.nn as nn


# localization approach proposed by
# Kohlbrenner, Maattrimilian, et al. "Towards best practice in eattrplaining neural network decisions with LRP."
# 2020 International Joint Conference on Neural Networks (IJCNN). IEEE, 2020.
# mu = Rin / Rout
# mu_w =  mu * (Stot / Sin )
class Localization(nn.Module):
    def __init__(self, f: nn.Module = None, exp: nn.Module = None, name: str = "loc"):
        super(Localization, self).__init__()

        self.name = name

    @torch.no_grad()
    def forward(self, x, attr, mask):

        attr = torch.einsum( "bmn,bn->bmn", attr, torch.sign(x))
        attr = nn.functional.relu(attr)

        Rtot = attr.sum( -1 ) # b, m
        Rtot = torch.where(Rtot > 0, Rtot, float("inf"))

        Rin = torch.einsum("bmn,bn->bm", attr, mask.float())
        mu = torch.einsum("bm,bm->bm", Rin, 1 / Rtot)

        Stot = torch.ones_like( mask ).sum( -1 ) # b
        Sin = mask.sum( -1 )

        return torch.einsum("bm,b->bm", mu, (Stot / Sin))
