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


def show_zoom(validation, results_by_name, index, size=48, title=None):
    """Enlarged crop around the hole of one image, without interpolation.

    The crop is centred on the eroded hole, so it shows filled texture
    rather than the seam. Nearest-neighbour display keeps pixel-scale
    artifacts (grids, blocks) visible instead of smoothing them.
    """
    interior = validation.interior[index, ..., 0] > 0.5
    rows, cols = np.nonzero(interior if interior.any() else validation.masks[index, ..., 0])
    height, width = interior.shape
    top = int(np.clip(round(rows.mean() - size / 2), 0, height - size))
    left = int(np.clip(round(cols.mean() - size / 2), 0, width - size))

    columns = [("Original", validation.originals), *results_by_name.items()]
    fig, axes = plt.subplots(1, len(columns), figsize=(3 * len(columns), 3.3), squeeze=False)
    for ax, (name, images) in zip(axes[0], columns):
        crop = images[index, top:top + size, left:left + size]
        ax.imshow(np.clip(crop, 0, 1), interpolation="nearest")
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
