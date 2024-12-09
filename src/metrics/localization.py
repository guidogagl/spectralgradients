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
        attr = nn.functional.relu(attr)

        Rtot = attr.reshape(attr.shape[0], -1).sum(1)

        Rtot = torch.where(Rtot > 0, Rtot, float("inf"))

        Rin = torch.einsum("bmn,bn->bm", attr, mask.float())
        mu = torch.einsum("bm,b->bm", Rin, 1 / Rtot)

        Stot = attr.shape[1:]
        Stot = torch.tensor(Stot).prod().item()

        Sin = mask.reshape(mask.shape[0], -1).sum(1)

        Sin = torch.where(Sin > 0, Sin, float("inf"))

        return torch.einsum("bm,b->bm", mu, (Stot / Sin))
