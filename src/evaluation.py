import torch.nn as nn
import matplotlib.pyplot as plt
import lpips
import torch
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as ssim

EVALUATION_REGISTRY = {}

def register_eval(name):
    """Decorator to register an evaluation metric function in the EVALUATION_REGISTRY."""
    def decorator(func):
        EVALUATION_REGISTRY[name] = func
        return func
    return decorator

def create_evaluator(metric_names):
    """Creates a callable evaluator that computes selected metrics for a model's predictions."""
    def evaluate(model, input, target) -> dict:
        results = {}
        prediction = model(input)
        # potential arguments needed by evaluation methods
        eval_args = {
            "model": model,
            "input": input,
            "target": target,
            "prediction": prediction
        }
        for name in metric_names:
            metric_fn = EVALUATION_REGISTRY[name]
            if metric_fn is None:
                raise ValueError(f"Unknown metric: {name}")
            results[name] = metric_fn(**eval_args)
        return results
    return evaluate

@register_eval("LPIPS+KLD")
def lpips_kld_combine(prediction, model, input, target, **kwargs):
    lp_val = lpips_fn(prediction, target)
    kld = kld_fn(model, input)
    return lp_val + 0.1 * kld

@register_eval("MSE_LPIPS")
def mse_lpips_loss_fn(prediction, target, alpha=1.0, beta=0.1, **kwargs):
    """
    Combined loss = alpha * MSE + beta * LPIPS
    """
    mse = mse_loss_fn(prediction, target, **kwargs)
    lpips = lpips_fn(prediction, target, **kwargs)
    return alpha * mse + beta * lpips

@register_eval("LPIPS")
def lpips_fn(prediction, target, **kwargs):
    if not hasattr(lpips_fn, "model"):
        lpips_fn.model = lpips.LPIPS(net='alex').to(prediction.device)
        lpips_fn.model.eval()

    return lpips_fn.model(prediction, target).mean()

@register_eval("PSNR")
def psnr_fn(prediction, target, **kwargs):
    mse = torch.mean((prediction - target) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    max_pixel = 1.0  # assuming inputs are normalized [0,1]; adjust if different scale
    psnr = 10 * torch.log10(max_pixel**2 / mse)
    return psnr

@register_eval("SSIM")
def ssim_fn(prediction, target, **kwargs):
    # Assumes prediction and target are normalized to [0,1] and shape (N, C, H, W)
    return ssim(prediction, target, data_range=1.0)

@register_eval("MSE")
def mse_loss_fn(prediction, target, **kwargs):
    criterion = nn.MSELoss()  
    return criterion(prediction, target)

@register_eval("KLD")
def kld_fn(model, input, **kwargs):
    # Ensure the model returns latents
    _, mu, logvar = model(input, return_latents=True)

    # Compute KL divergence per batch (averaged per pixel)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])

    return kld.mean()

@register_eval("plot_model_reconstruction")
def plot_model_reconstruction(prediction, input, **kwargs):
    """ Return a matplotlib Figure comparing input image and reconstruction."""
    input = input.cpu().detach().numpy().squeeze(0)
    prediction = prediction.cpu().detach().numpy().squeeze(0)
    print(input.shape, prediction.shape)
   
    if input.ndim == 3 and input.shape[0] == 3:  
        input = input.transpose(1, 2, 0)  
    if prediction.ndim == 3 and prediction.shape[0] == 3:
        prediction = prediction.transpose(1, 2, 0)
        
    fig, axs = plt.subplots(1, 2, figsize=(2, 1))     

    axs[0].imshow(input)
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    axs[1].imshow(prediction)
    axs[1].set_title("Reconstruction")
    axs[1].axis("off")

    plt.tight_layout()
    return fig