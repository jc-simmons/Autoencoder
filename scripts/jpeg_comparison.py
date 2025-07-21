""" Benchmark JPEG reconstruction on CIFAR-10 images and report average distortion metrics and compression ratio."""

import io
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

from src.dataset import create_datasets
from src.evaluation import create_evaluator

to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

def jpeg_reconstruct(image_tensor, quality=50):
    """Compresses and decompresses an image tensor using JPEG at the given quality, 
    returning the reconstructed image and compressed size in bytes."""
    img_pil = to_pil(image_tensor.squeeze(0)) 
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    compressed_size = len(buffer.getvalue())
    buffer.seek(0)
    img_jpeg = Image.open(buffer).convert("RGB")
    return img_jpeg, compressed_size

def compute_avg(metrics_list):
    keys = metrics_list[0].keys()
    return {k: np.mean([m[k] for m in metrics_list]) for k in keys}

def main():
    _, val_data = create_datasets(data_path=Path('data/CIFAR10/'))
    eval_subset = torch.utils.data.Subset(val_data, range(10000))
    eval_loader = torch.utils.data.DataLoader(eval_subset, batch_size=1, shuffle=False)

    jpeg_quality = 60
    results = {"metrics": [], "compression_ratio": []}
    evaluate_metrics = create_evaluator(['MSE', 'PSNR', 'SSIM'])

    for img, _ in eval_loader:
        recon_jpeg, jpeg_size = jpeg_reconstruct(img, quality=jpeg_quality)
        arr_uint8 = np.array(recon_jpeg)
        assert arr_uint8.dtype == np.uint8
        jpeg_metrics = evaluate_metrics(None, to_tensor(recon_jpeg).unsqueeze(0), img)
        results["metrics"].append(jpeg_metrics)
        results["compression_ratio"].append(arr_uint8.nbytes / jpeg_size)

    avg_jpeg_metrics = compute_avg(results["metrics"])
    avg_jpeg_ratio = np.mean(results["compression_ratio"])

    print("Average Metrics:", avg_jpeg_metrics)
    print("Average Compression Ratio:", avg_jpeg_ratio)

if __name__ == "__main__":
    main()