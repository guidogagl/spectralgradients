import torch
from torch import nn

@torch.no_grad()
def step_integral(
    fn: nn.Module,
    t: torch.Tensor,
    i: int,
    path: callable,
    jac_path: callable = None,
    device: torch.device = "cpu",
):
    G0 = torch.func.vmap(torch.func.jacrev(fn))(path(t[i - 1]))
    G1 = torch.func.vmap(torch.func.jacrev(fn))(path(t[i]))

    if jac_path is None:
        Jp0 = torch.func.vmap(torch.func.jacfwd(path))(
            torch.ones(1, device=device) * t[i - 1]
        ).squeeze()
        Jp1 = torch.func.vmap(torch.func.jacfwd(path))(
            torch.ones(1, device=device) * t[i]
        ).squeeze()
    else:
        Jp0 = jac_path(t[i - 1])
        Jp1 = jac_path(t[i])

    Jp0, Jp1 = torch.abs(Jp0), torch.abs(Jp1)

    G0 = torch.einsum("bmn,bn->bmn", G0, Jp0)
    G1 = torch.einsum("bmn,bn->bmn", G0, Jp1)

    SG = (t[i] - t[i - 1]) * (G0 + G1) / 2

    return SG

@torch.no_grad()
def path_integral(
    fn: nn.Module,
    path: callable,
    jac_path: callable = None,
    n_points: int = 50,
):

    p = path(0)

    T = torch.linspace(0, 1, n_points, device=path(0).device, dtype=p.dtype)

    SG = 0

    curve = []

    for t in T:

        G = torch.func.vmap(torch.func.jacrev(fn))(path(t))

        if jac_path is None:
            J = torch.func.vmap(torch.func.jacfwd(path))(
                torch.ones(1, device=p.device) * t
            ).squeeze()
        else:
            J = jac_path(t)

        # J = torch.abs(J)

        G = torch.einsum("bmn,bn->bmn", G, J)

        curve = curve + [G]

    curve = torch.stack(curve, dim=0)
    curve = torch.trapezoid(curve, dx=1 / (n_points - 1), dim=0)

    return curve


def line_integral(
    fn: nn.Module,
    x_1: torch.Tensor,
    x_0: torch.Tensor,
    n_points: int = 50,
):
    def path(alpha: torch.Tensor):
        return x_1 * alpha - (1 - alpha) * x_0

    return path_integral(fn=fn, path=path, jac_path=None, n_points=n_points)


def fline_integral(
    fn: nn.Module,
    x_1: torch.Tensor,
    x_0: torch.Tensor,
    n_points: int = 50,
):
    def path(alpha: torch.Tensor):
        return x_0 - alpha * (x_0 - x_1)

    def jac_path(alpha: torch.Tensor):
        return -(x_0 - x_1)

    return path_integral(fn=fn, path=path, jac_path=jac_path, n_points=n_points)
