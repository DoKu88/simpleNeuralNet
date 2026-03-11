import os
import sys
import subprocess
from datetime import datetime
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import petname

from nodes import Convolution2DLayer, SquaredLoss


# =============================================================================
# Dataset: simple synthetic grayscale images + salt-and-pepper noise
# =============================================================================

def generate_base_images(
    n_images: int = 64,
    height: int = 32,
    width: int = 32,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a small set of simple synthetic grayscale images.
    Each image is a combination of a smooth background and a few random blobs.

    Returns:
        images: array of shape (n_images, height, width) with values in [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng()

    images = np.zeros((n_images, height, width), dtype=float)

    xs = np.linspace(0, 2 * np.pi, width)
    ys = np.linspace(0, 2 * np.pi, height)
    X, Y = np.meshgrid(xs, ys)

    for i in range(n_images):
        # Smooth background: sinusoidal pattern
        freq_x = rng.uniform(0.5, 2.0)
        freq_y = rng.uniform(0.5, 2.0)
        phase_x = rng.uniform(0, 2 * np.pi)
        phase_y = rng.uniform(0, 2 * np.pi)

        base = 0.5 + 0.25 * np.sin(freq_x * X + phase_x) * np.cos(freq_y * Y + phase_y)

        # Add a few random bright/dark blobs
        img = base
        num_blobs = rng.integers(2, 6)
        for _ in range(num_blobs):
            cx = rng.integers(0, width)
            cy = rng.integers(0, height)
            radius = rng.integers(height // 8, height // 4)
            yy, xx = np.ogrid[:height, :width]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
            blob_intensity = rng.choice([0.0, 1.0])
            img = np.where(mask, blob_intensity, img)

        img = np.clip(img, 0.0, 1.0)
        images[i] = img

    return images


def add_salt_and_pepper_noise_2d(
    images: np.ndarray,
    noise_prob: float = 0.05,
    salt_vs_pepper: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Add salt-and-pepper noise to a batch of grayscale images.

    Args:
        images: array of shape (N, H, W) with values in [0, 1].
        noise_prob: probability that any given pixel is corrupted.
        salt_vs_pepper: fraction of corrupted pixels that become "salt" (1.0);
                        rest become "pepper" (0.0).

    Returns:
        noisy_images: array of same shape as `images`.
    """
    if rng is None:
        rng = np.random.default_rng()

    noisy_images = images.copy()
    N, H, W = noisy_images.shape

    # Decide which pixels get corrupted
    rand_vals = rng.random(size=(N, H, W))
    corrupt_mask = rand_vals < noise_prob

    # Among corrupted pixels, decide salt vs pepper
    salt_mask = rng.random(size=(N, H, W)) < salt_vs_pepper
    salt_mask = np.logical_and(corrupt_mask, salt_mask)
    pepper_mask = np.logical_and(corrupt_mask, ~salt_mask)

    noisy_images[salt_mask] = 1.0
    noisy_images[pepper_mask] = 0.0

    return noisy_images


# =============================================================================
# Visualization helpers (similar spirit to 1D_convolution.py)
# =============================================================================

def save_epoch_frame_2d(
    clean_image: np.ndarray,
    noisy_image: np.ndarray,
    denoised_image: np.ndarray,
    kernel: np.ndarray,
    loss_value: float,
    epoch_idx: int,
    frames_dir: str,
) -> None:
    """
    Save a matplotlib figure showing:
      - clean image
      - noisy image
      - denoised (predicted) image
      - current convolution kernel
    for a given epoch.
    """
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    ax0, ax1, ax2, ax3 = axes

    im0 = ax0.imshow(clean_image, cmap="gray", vmin=0.0, vmax=1.0)
    ax0.set_title("Clean")
    ax0.axis("off")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    im1 = ax1.imshow(noisy_image, cmap="gray", vmin=0.0, vmax=1.0)
    ax1.set_title("Noisy")
    ax1.axis("off")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(denoised_image, cmap="gray", vmin=0.0, vmax=1.0)
    ax2.set_title("Denoised")
    ax2.axis("off")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    im3 = ax3.imshow(kernel, cmap="bwr")
    ax3.set_title("Kernel")
    ax3.axis("off")
    for (i, j), val in np.ndenumerate(kernel):
        ax3.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color="black")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    fig.suptitle(f"Epoch {epoch_idx + 1}, loss={loss_value:.4f}")
    plt.tight_layout()
    frame_path = os.path.join(frames_dir, f"frame_{epoch_idx:04d}.png")
    fig.savefig(frame_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_loss_curve_2d(
    loss_history: List[float],
    epochs: int,
    output_dir: str,
    run_ts: str,
    run_name: str,
) -> None:
    """
    Plot loss over epochs and save as an image into `output_dir`.
    """
    epochs_arr = np.arange(1, epochs + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs_arr, loss_history, linewidth=0.8, marker="o", markersize=2)
    plt.xlabel("Epoch")
    plt.ylabel("Average loss")
    plt.title("2D Conv Denoising: Training loss over epochs")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_plot_path = os.path.join(output_dir, f"{run_ts}_{run_name}_2d_conv_loss_curve.png")
    plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_video_2d(
    frames_dir: str,
    output_dir: str,
    run_ts: str,
    run_name: str,
    seconds_per_frame: float = 0.5,
) -> None:
    """
    Create an MP4 video from PNG frames stored in `frames_dir`.

    Frames are expected to be named like 'frame_XXXX.png'. The video is saved
    into `output_dir` as '[run_ts]_[run_name]_conv_training.mp4'.
    """
    frame_files = sorted(
        f for f in os.listdir(frames_dir) if f.endswith(".png") and f.startswith("frame_")
    )
    if not frame_files:
        return

    video_path = os.path.join(output_dir, f"{run_ts}_{run_name}_conv_training.mp4")
    input_fps = 1.0 / seconds_per_frame
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(input_fps),
            "-i",
            os.path.join(frames_dir, "frame_%04d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            video_path,
        ],
        check=True,
    )

    for fname in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, fname))
    os.rmdir(frames_dir)


# =============================================================================
# Simple 2D CNN denoising model using nodes.Convolution2DLayer + SquaredLoss
# =============================================================================

def train_conv2d_denoiser(
    epochs: int = 100,
    n_images: int = 64,
    image_size: Tuple[int, int] = (32, 32),
    kernel_size: int = 5,
    learning_rate: float = 1e-4,
    noise_prob: float = 0.1,
    salt_vs_pepper: float = 0.5,
) -> None:
    """
    Train a simple 2D convolutional denoiser on synthetic images with salt-and-pepper noise,
    using the same node-based pattern as in `neural_net_wi_autograd.py`.
    """
    H, W = image_size
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")
    padding = (kernel_size - 1) // 2
    rng = np.random.default_rng()

    clean_images = generate_base_images(
        n_images=n_images, height=image_size[0], width=image_size[1], rng=rng
    )

    # Set up run identifiers and output directories
    run_ts = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    run_name = petname.Generate(2, "_")
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, f"{run_ts}_{run_name}_conv2d_video_tmp")
    os.makedirs(frames_dir, exist_ok=True)

    # Build a simple stackable conv network: multiple conv layers with same
    # input/output spatial shape, followed by a SquaredLoss node.
    in_features = np.array([H, W])
    out_features = np.array([H, W])

    conv2D_1 = Convolution2DLayer(
            kernel_size=np.array([kernel_size, kernel_size]),
            in_features=in_features,
            out_features=out_features,
            padding_x=padding,
            padding_y=padding,
        )
      
    conv2D_2 = Convolution2DLayer(
            kernel_size=np.array([kernel_size, kernel_size]),
            in_features=in_features,
            out_features=out_features,
            padding_x=padding,
            padding_y=padding,
        )
    conv2D_3 = Convolution2DLayer(
            kernel_size=np.array([kernel_size, kernel_size]),
            in_features=in_features,
            out_features=out_features,
            padding_x=padding,
            padding_y=padding,
        )

    loss_node = SquaredLoss()
    nodes: List[object] = [conv2D_1, conv2D_2, conv2D_3, loss_node]
    layers, loss_node = nodes[:-1], nodes[-1]

    loss_history: List[float] = []

    # Training loop
    for epoch in range(epochs):
        # New noisy version each epoch (like 1D example)
        noisy_images = add_salt_and_pepper_noise_2d(
            clean_images,
            noise_prob=noise_prob,
            salt_vs_pepper=salt_vs_pepper,
            rng=rng,
        )

        epoch_loss = 0.0

        for idx in range(n_images):
            clean = clean_images[idx]
            noisy = noisy_images[idx]

            h = noisy
            for node in layers:
                h = node.forward(h)

            # Squared loss between clean and denoised; use mean per pixel
            loss_value = loss_node.forward(h.ravel(), clean.ravel()) / (H * W)
            epoch_loss += loss_value

            # Backward: loss then layers in reverse
            upstream = loss_node.backward(1.0) / (H * W)  # shape (H*W,)
            upstream_image = upstream.reshape(image_size)
            for node in reversed(layers):
                upstream_image = node.backward(upstream_image)

            # Gradient descent for all nodes with parameters
            for node in nodes:
                node.update_weights(learning_rate)

        avg_loss = epoch_loss / n_images
        loss_history.append(avg_loss)
        print(f"Epoch {epoch} / {epochs}  |  avg loss = {avg_loss:.6f}")

        # Visualization of the first image in the dataset for this epoch
        idx_plot = 0
        clean_plot = clean_images[idx_plot]
        noisy_plot = noisy_images[idx_plot]
        h_plot = noisy_plot
        for node in layers:
            if node is loss_node:
                break
            h_plot = node.forward(h_plot)
        denoised_plot = h_plot
        save_epoch_frame_2d(
            clean_image=clean_plot,
            noisy_image=noisy_plot,
            denoised_image=denoised_plot,
            kernel=conv2D_1.kernel,
            loss_value=avg_loss,
            epoch_idx=epoch,
            frames_dir=frames_dir,
        )

    # Save loss curve and create training video
    save_loss_curve_2d(loss_history, epochs, output_dir, run_ts, run_name)
    create_video_2d(frames_dir, output_dir, run_ts, run_name, seconds_per_frame=0.25)


def main() -> None:
    """
    Entry point: constructs the simple 2D CNN denoiser and trains it to denoise images.
    """
    # You can tweak these hyperparameters if desired.
    train_conv2d_denoiser(
        epochs=100,
        n_images=500,
        image_size=(32, 32),
        kernel_size=5,  # odd-sized kernel so that padding = (k-1)/2 is integer
        learning_rate=1e-2,
        noise_prob=0.1,
        salt_vs_pepper=0.5,
    )


if __name__ == "__main__":
    main()

