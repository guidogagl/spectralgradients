import torch
from torch import nn

from tqdm.autonotebook import tqdm
import gc

def gaussian_perturbation(x: torch.Tensor, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)

    I = torch.randn_like( x )

    return x + (.1*I)


class LocalLipschitzEstimate(nn.Module):
    def __init__(
        self,
        f: callable,  # f is the explanation function, not the model
        I: callable = gaussian_perturbation,  # value to substitute the feature with
        n_trials: int = 50,
        similarity: callable = torch.cdist,
    ):

        super(LocalLipschitzEstimate, self).__init__()
        self.f = f
        self.I = I
        self.similarity = similarity
        self.n_trials = n_trials

    @torch.no_grad()
    def forward(self, x, attr, mask):

        # computes the LLE for x and attr
        # LLE = E[ || f(x) - F(x_I) ||_2 / ||x - x_I||_2 ]

        batch_size, m, n = attr.shape

        lle = torch.zeros(batch_size, m).to(x.device)

        attr = attr.reshape(batch_size * m, n)

        for i in range(self.n_trials) :

            x_I = self.I(x, seed=i)
            Den = 1 / self.similarity(x.unsqueeze(1), x_I.unsqueeze(1)).reshape(
                batch_size
            )


            Num = self.f(x_I)
            Num = Num.reshape(batch_size * m, 1, n)
            Num = self.similarity(attr.unsqueeze(1), Num).reshape(
                batch_size, m
            )  # b, m , n

            lle = torch.maximum(lle, torch.einsum("bm,b->bm", Num, Den))

        return lle