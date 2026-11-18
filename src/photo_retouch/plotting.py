"""Side-by-side visual comparisons."""
import matplotlib.pyplot as plt
import numpy as np

from .metrics import masked_mae_per_image


def show_comparison(validation, results_by_name, indices, title=None):
    """Original, masked input, then one column per method, with per-image MAE."""
    columns = [
        ("Original", validation.originals),
        ("Masked input", validation.visible),
        *results_by_name.items(),
    ]
    column_mae = {
        name: masked_mae_per_image(images, validation.originals, validation.masks)
        for name, images in results_by_name.items()
    }
    fig, axes = plt.subplots(
        len(indices), len(columns),
        figsize=(3 * len(columns), 3 * len(indices)), squeeze=False,
    )
    for row, index in enumerate(indices):
        for col, (name, images) in enumerate(columns):
            ax = axes[row, col]
            ax.imshow(np.clip(images[index], 0, 1))
            label = name
            if name in column_mae:
                label += f"\nMAE {column_mae[name][index]:.3f}"
            ax.set_title(label, fontsize=9)
            ax.axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
