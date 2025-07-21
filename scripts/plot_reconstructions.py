"""Simple eval script for plotting original and autoencoded images for given dataset indices."""

import matplotlib.pyplot as plt
from pathlib import Path

from src.dataset import create_datasets
from src.torch_utils import load_torch_model, Stage
from src.model import CAE


def plot_reconstruction_grid(img_pairs, grid):
    """Plots original and reconstructed image pairs in a side-by-side grid layout."""
    rows, cols = grid
    assert cols % 2 == 0, "Number of columns must be even (pairs of images)."
    pairs_per_row = cols // 2
    total_pairs = len(img_pairs)
    max_pairs = rows * pairs_per_row
    assert total_pairs <= max_pairs, f"Grid too small for {total_pairs} pairs."

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)

    for idx, (orig, recon) in enumerate(img_pairs):
        row = idx // pairs_per_row
        col_img = (idx % pairs_per_row) * 2
        col_recon = col_img + 1

        axes[row, col_img].imshow(orig)
        axes[row, col_img].axis('off')
        if row == 0:
            axes[row, col_img].set_title("Original", fontsize=12)

        axes[row, col_recon].imshow(recon)
        axes[row, col_recon].axis('off')
        if row == 0:
            axes[row, col_recon].set_title("Reconstructed", fontsize=12)

    # Hide unused axes
    for ax_row in range(rows):
        for ax_col in range(cols):
            pair_idx = ax_row * pairs_per_row + (ax_col // 2)
            if pair_idx >= total_pairs:
                axes[ax_row, ax_col].axis('off')

    plt.tight_layout()
    return fig

def main(indices, grid):
    """Simple eval script for plotting original and autoencoded images for given dataset indices."""
    dataset_kwargs= {"data_path": Path('data/CIFAR10/')}
    _, val_data = create_datasets(**dataset_kwargs)

    model_kwargs={"latent_channels": 12, "hidden_channels": [64, 128]}
    model = load_torch_model(CAE(**model_kwargs), Path('output/test/model.pt'))

    images_to_plot = []
    
    with Stage.VAL(model):
        for i in indices:
            img, _ = val_data[i]
            recon = model(img.unsqueeze(0))

            img_np = img.permute(1, 2, 0).cpu().numpy()
            recon_np = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()

            images_to_plot.append([img_np, recon_np])

    img_plot = plot_reconstruction_grid(images_to_plot, grid)
    return img_plot

if __name__ == "__main__":
    indices = [0,1,2,3,4,5,6,7,8]
    grid = (3,6)
    main(indices, grid)
    plt.show()