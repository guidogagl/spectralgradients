import torch
import gc


from src.posthoc.egradients import ExpectedGradients
from src.posthoc.igradients import IntegratedGradients
from src.posthoc.saliency import Saliency, InputXGradient
from src.posthoc.sgradients import SpectralGradients

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def explainer_wrapper(**kwargs):
    """Wrapper for explainer functions."""
    if kwargs["method"] == "Saliency":
        return saliency_explainer(**kwargs)
    elif kwargs["method"] == "IntegratedGradients":
        return intgrad_explainer(**kwargs)
    elif kwargs["method"] == "ExpectedGradients":
        return expgrad_explainer(**kwargs)
    elif kwargs["method"] == "InputXGrad":
        return inputxgrad_explainer(**kwargs)
    elif kwargs["method"] == "SpectralGradients":
        return spectralgrads_explainer(**kwargs)
    elif kwargs["method"] == "WSpectralGradients":
        return wspectralgrads_explainer(**kwargs)
    else:
        raise ValueError("Pick an explaination function that exists.")


def saliency_explainer(model, inputs, targets, **kwargs):

    gc.collect()
    torch.cuda.empty_cache()

    saliency = Saliency(fn=model).to(device)

    explanations = saliency(inputs.to(device))
    # explanations shape : (batch_size, num_classes, num_features)
    # select target class for each sample, targets shape: (batch_size,)
    explanations = explanations.detach().cpu()

    return saliency.to("cpu"), explanations


def inputxgrad_explainer(model, inputs, targets, **kwargs):

    gc.collect()
    torch.cuda.empty_cache()

    inputxgrad = InputXGradient(fn=model).to(device)

    explanations = inputxgrad(inputs.to(device))
    # explanations shape : (batch_size, num_classes, num_features)
    # select target class for each sample, targets shape: (batch_size,)
    explanations = explanations.detach().cpu()

    return inputxgrad.to("cpu"), explanations


def intgrad_explainer(
    model, inputs, targets, baseline: float = 0.0, n_points: int = 50, **kwargs
):

    gc.collect()
    torch.cuda.empty_cache()

    intgrad = IntegratedGradients(
        fn=model, baselines=torch.zeros_like(inputs), n_points=n_points
    ).to(device)

    explanations = intgrad(inputs.to(device))
    # explanations shape : (batch_size, num_classes, num_features)
    # select target class for each sample, targets shape: (batch_size,)
    explanations = explanations.detach().cpu()

    return intgrad.to("cpu"), explanations


def expgrad_explainer(
    model, inputs, targets, baseline: torch.Tensor, n_points: int = 50, **kwargs
):

    gc.collect()
    torch.cuda.empty_cache()

    intgrad = ExpectedGradients(fn=model, baselines=baseline, n_points=n_points).to(
        device
    )

    explanations = intgrad(inputs.to(device))
    # explanations shape : (batch_size, num_classes, num_features)
    # select target class for each sample, targets shape: (batch_size,)
    explanations = explanations.detach().cpu()

    return intgrad.to("cpu"), explanations


def spectralgrads_explainer(
    model, inputs, targets, fs: int = 100, Q=5, nperseg=200, noverlap=100, **kwargs
):

    gc.collect()
    torch.cuda.empty_cache()

    spectralgrads = SpectralGradients(
        fn=model, fs=fs, Q=Q, nperseg=nperseg, noverlap=noverlap
    ).to(device)

    explanations = spectralgrads(inputs.to(device))

    # explanations shape : (batch_size, num_classes, frequencies, num_features)
    explanations = explanations.sum(dim=2)
    explanations = explanations.detach().cpu()

    return spectralgrads.to("cpu"), explanations


def wspectralgrads_explainer(
    model, inputs, targets, fs: int = 100, Q=5, nperseg=200, noverlap=100, **kwargs
):

    gc.collect()
    torch.cuda.empty_cache()

    spectralgrads = SpectralGradients(
        fn=model, fs=fs, Q=Q, nperseg=nperseg, noverlap=noverlap
    ).to(device)

    explanations = spectralgrads(inputs.to(device))

    # explanations shape : (batch_size, num_classes, frequencies, num_features)
    # first compute the weights
    weights = torch.abs(explanations.sum(dim=-1))  # frequency weights

    explanations = explanations * weights.unsqueeze(-1)
    explanations = explanations.sum(dim=2)

    explanations = explanations.detach().cpu()

    return spectralgrads.to("cpu"), explanations


from src.metrics.lle import LocalLipschitzEstimate as LLE
from src.metrics.infidelity import IROF
from src.metrics.localization import Localization


def run_eval(metric, metric_kwargs, x, attr, mask):
    gc.collect()
    torch.cuda.empty_cache()

    metric_ = metric(**metric_kwargs).to(device)

    l1 = metric_(x.to(device), attr.to(device), mask.to(device)).detach().cpu()

    return l1


def localization(explainer, model, x, y, attr, mask):

    # we need to avoid the sample with all-zeros mask
    batch_size = x.shape[0]
    tmp_mask, tmp_attr, tmp_x = [], [], []

    for i in range(batch_size):
        if mask[i].sum() == 0:
            continue

        tmp_attr += [attr[i, y[i].long()]]
        tmp_mask += [mask[i]]
        tmp_x += [x[i]]

    tmp_attr = torch.stack(tmp_attr)
    tmp_x = torch.stack(tmp_x)
    tmp_mask = torch.stack(tmp_mask)

    return run_eval(Localization, {}, tmp_x, tmp_attr, tmp_mask).mean()


def lle(explainer, model, x, y, attr, mask):
    lle_result = run_eval(LLE, {"f": explainer}, x, attr, mask)

    result = torch.zeros(x.shape[0])

    for i in range(lle_result.shape[0]):
        result[i] = lle_result[i, y[i].long().item()]

    return result.mean()


def irof(explainer, model, x, y, attr, mask):
    irof_result = run_eval(
        IROF,
        {
            "f": model,
            "n_features": 1,
        },
        x,
        attr,
        mask,
    )

    result = torch.zeros(len(irof_result))
    for i in range(irof_result.shape[0]):
        result[i] = irof_result[i, y[i].long().item()]

    # return result.mean()
    return result.mean()
