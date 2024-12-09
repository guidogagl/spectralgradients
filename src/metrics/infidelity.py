import torch
import torch.nn as nn

from tqdm.autonotebook import tqdm


class Infidelity(nn.Module):
    def __init__(
        self,
        f: callable,
        patch: float = 0,  # value to substitute the feature with
        percentage_res: float = 0.1,  # number of feature to remove at each step
        name: str = "inf",
        **kwargs,
    ):
        super(Infidelity, self).__init__()
        self.f = f
        self.patch = patch
        self.percentages = torch.arange(percentage_res, 1, percentage_res)
        self.name = name

    @torch.no_grad()
    def forward(self, x, attr, mask):
        attr_ = attr.clone()
        # pooling the attributions
        patch_size = int(self.percentages[0] * x.size(-1))

        for i in range(0, x.size(-1), patch_size):
            max_pool, _ = attr[:, :, i : i + patch_size].max(dim=-1)
            max_pool = max_pool.unsqueeze(-1).repeat(1, 1, patch_size)
            attr_[:, :, i : i + patch_size] = max_pool

        attr = attr_
        sorted_attr = torch.argsort(attr, dim=-1)

        inf = [self.f(x).cpu()]

        for i, perc in enumerate(self.percentages):

            occlusion_attr = sorted_attr[..., i * patch_size : (i + 1) * patch_size]

            for b in range(0, x.size(0)):
                x[b, occlusion_attr[b].long()] = self.patch

            inf = inf + [self.f(x).cpu()]

        inf = inf + [self.f(torch.ones_like(x) * self.patch).cpu()]

        inf = torch.stack(inf, dim=0).to(x.device)
        inf = torch.einsum("pbm,bm->pbm", inf, 1 / inf[0])
        inf = torch.trapezoid(inf, dx=1 / inf.shape[0], dim=0)

        return inf
