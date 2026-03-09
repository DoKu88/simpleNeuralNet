import os
import sys
from pprint import pprint
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import petname

# =============================================================================
# GENERATE DATASET 
# =============================================================================
def generate_random_sin_cos_functions(n=100, x_min=0, x_max=2 * np.pi, num_points=100):
    """
    Generate n random sin/cos functions with random amplitudes and periods.
    Returns:
        funcs: list of callable functions f(x)
        params: list of (type, amplitude, frequency, phase) tuples
        xs: the x values (array of shape (num_points,))
        ys: array shape (n, num_points), holding function values for each function
    """

    funcs = []
    params = []
    ys = []
    xs = np.linspace(x_min, x_max, num_points)
    rng = np.random.default_rng()
    for _ in range(n):
        func_type = rng.choice(["sin", "cos"])
        amplitude = rng.uniform(0.5, 3.0)
        frequency = rng.uniform(0.5, 3.0)
        phase = rng.uniform(0, 2 * np.pi)
        if func_type == "sin":
            f = lambda x, a=amplitude, w=frequency, p=phase: a * np.sin(w * x + p)
        else:
            f = lambda x, a=amplitude, w=frequency, p=phase: a * np.cos(w * x + p)
        funcs.append(f)
        params.append((func_type, amplitude, frequency, phase))
        ys.append(f(xs))
    ys = np.array(ys)
    return funcs, params, xs, ys


def add_noise_to_function(xs, ys, noise_std=0.1, rng=None):
    """
    Add Gaussian noise to an existing sampled function.

    Args:
        xs: 1D NumPy array of x locations (not modified, included for convenience).
        ys: 1D NumPy array of function values at `xs`.
        noise_std: Standard deviation of the additive Gaussian noise.
        rng: Optional NumPy Generator. If None, a default generator is used.

    Returns:
        noisy_ys: Function values with added noise, same shape as `ys`.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(loc=0.0, scale=noise_std, size=ys.shape)
    noisy_ys = ys + noise
    return noisy_ys


def plot_function(xs, ys, labels=None):
    """
    Plot one or more 1D curves given xs and ys.

    Args:
        xs: 1D NumPy array of x locations.
        ys: 1D NumPy array of y values or a list/tuple of such arrays.
        labels: Optional label or list of labels matching `ys`.
    """
    plt.figure()
    if isinstance(ys, (list, tuple)):
        for i, y in enumerate(ys):
            if labels is None:
                label = f"f{i}(x)"
            elif isinstance(labels, (list, tuple)):
                label = labels[i] if i < len(labels) else f"f{i}(x)"
            else:
                label = str(labels)
            plt.plot(xs, y, label=label)
    else:
        label = labels if labels is not None else "f(x)"
        plt.plot(xs, ys, label=label)

    plt.xlabel("x")
    plt.ylabel("f(x)")
    if labels is not None:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# =============================================================================
# Training and Optimizing 1D Convolution Kernel 
# =============================================================================

def loss_funct(ground_truth, predicted):
  # Squared loss, take the dot product of the difference between ground truth & predicted
  loss = (ground_truth - predicted) @ (ground_truth - predicted)
  return loss 

def loss_gradient(ground_truth, predicted):
  loss_gradient = -2 * (ground_truth - predicted)
  return loss_gradient

def loss_convolution_kern_j(input_z, padding_p, predicted_y, kernel_idx):
    # return vector of each dy_i/dw_j for all y_i's wrt a chosen w_j
    # from the blog post we know: dy_i(w_j) = z(i-j + p) since 0 indexed don't include +1
    dy_dwj = [0 for i in range(len(predicted_y))]
    for i in range(len(predicted_y)):
      if int(i - kernel_idx + padding_p) >= len(input_z) or int(i - kernel_idx + padding_p) < 0:
        continue
      dy_dwj[i] = input_z[int(i - kernel_idx + padding_p)]
    np.asarray(dy_dwj)

    return dy_dwj

def predict_y(input_z, padding_p, w_kernel):
  # we know that len(pred_y) = len(input_z) thus 
  pred_y = [0 for i in range(len(input_z))]
  for i in range(len(pred_y)):
    for b in range(len(w_kernel)):
      # 0 indexed, so don't need +1 term from blog post
      if int(i - b + padding_p) >= len(input_z) or int(i - b + padding_p) < 0:
        continue
      pred_y[i] += w_kernel[b] * input_z[int(i - b + padding_p)] 

  return np.asarray(pred_y)

# =============================================================================
# Utility Functions
# =============================================================================
def save_epoch_frame(xs, ground_truth, predicted, input_z, w_kernel, loss_value, epoch_idx, frames_dir):
  """
  Save a matplotlib figure showing signals (input_z, ground_truth, predicted)
  and the current convolution kernel for a given epoch.
  """
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
  ax1.plot(xs, input_z, label="input_z (noisy)", alpha=0.6)
  ax1.plot(xs, ground_truth, label="ground_truth", linewidth=2)
  ax1.plot(xs, predicted, label="predicted_y", linewidth=2)
  ax1.set_ylabel("signal")
  ax1.set_title(f"Signals (epoch {epoch_idx + 1}, loss={loss_value:.4f})")
  ax1.legend()
  ax1.grid(True, alpha=0.3)

  ax2.stem(np.arange(len(w_kernel)), w_kernel)
  ax2.set_xlabel("kernel index")
  ax2.set_ylabel("weight")
  ax2.set_title("Convolution kernel")
  ax2.grid(True, alpha=0.3)

  plt.tight_layout()
  frame_path = os.path.join(frames_dir, f"frame_{epoch_idx:04d}.png")
  fig.savefig(frame_path, dpi=150, bbox_inches="tight")
  plt.close(fig)


def save_loss_curve(loss_history, epochs, frames_dir):
  """
  Plot loss over epochs and save as an image into `frames_dir`.
  """
  epochs_arr = np.arange(1, epochs + 1)
  plt.figure(figsize=(8, 4))
  plt.plot(epochs_arr, loss_history, marker="o")
  plt.xlabel("Epoch")
  plt.ylabel("Average loss")
  plt.title("Training loss over epochs")
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  loss_plot_path = os.path.join(frames_dir, "loss_curve.png")
  plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
  plt.close()

def create_video(frames_dir, output_dir, run_ts, run_name, fps=10, seconds_per_frame=1):
  """
  Create an MP4 video from PNG frames stored in `frames_dir`.

  Frames are expected to be named like 'frame_XXXX.png'. The video is saved
  into `output_dir` as '[run_ts]_[run_name]_conv_training.mp4'. Temporary
  frame images are deleted after the video is written.

  Each frame is duplicated (fps * seconds_per_frame) times so it stays visible
  for `seconds_per_frame` seconds. Using fps >= 10 avoids codec frame-dropping
  that occurs at very low frame rates (e.g. fps=1).
  """
  frame_files = sorted(
      f for f in os.listdir(frames_dir) if f.endswith(".png") and f.startswith("frame_")
  )
  if not frame_files:
    return

  first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
  if first_frame is None:
    return

  height, width, _ = first_frame.shape
  video_path = os.path.join(output_dir, f"{run_ts}_{run_name}_conv_training.mp4")
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

  repeats = max(1, int(fps * seconds_per_frame))
  for fname in frame_files:
    img = cv2.imread(os.path.join(frames_dir, fname))
    if img is None:
      continue
    for _ in range(repeats):
      writer.write(img)

  writer.release()

  #for fname in os.listdir(frames_dir):
  #  os.remove(os.path.join(frames_dir, fname))

# Example usage
if __name__ == "__main__":
    # Dataset generation 
    funcs, params, xs, ys = generate_random_sin_cos_functions(100)
    func_type, amp, freq, phase = params[0]
    print(f"{func_type}(x): amplitude={amp:.2f}, frequency={freq:.2f}, phase={phase:.2f}")
    noisy_ys = [add_noise_to_function(xs, ys[i], noise_std=0.2) for i in range(len(ys))]

    # Run identifier for saved outputs: [timestamp]_[RUN_NAME]_[type]
    run_ts = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    run_name = petname.Generate(2, "_")
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, f"{run_ts}_{run_name}_conv_video_tmp")
    os.makedirs(frames_dir, exist_ok=True)

    # Define kernel and padding 
    kernel_length = 5
    w_kernel = np.random.rand(kernel_length)
    padding_p = (len(w_kernel) -1) / 2 
    if padding_p > int(padding_p):
      print(f"padding_p {padding_p} not integer! check w_kernel length {w_kernel.shape} to be odd")
      sys.exit()

    # other ML specific params
    learning_rate = 1e-4

    # Training Loop 
    epochs = 10  # 50 
    loss_history = []
    for e in range(epochs):
      pred_ys = []
      for train_idx in range(len(xs)):
        ground_truth = ys[train_idx]
        input_z = noisy_ys[train_idx]
        pred_y = predict_y(input_z, padding_p, w_kernel)
        pred_ys.append(pred_y)

      # need to calculate all the dL_dw_i's 
      dL_dwj = np.zeros((len(w_kernel)))
      loss_avg = 0.0
      for idx in range(len(pred_ys)):
        loss_avg += loss_funct(ys[idx], pred_ys[idx])
        dL_dy = loss_gradient(ys[idx], pred_ys[idx])

        dy_dw = [[] for i in range(len(w_kernel))]
        for kern_idx_j in range(len(w_kernel)):
          dy_dw[kern_idx_j] = loss_convolution_kern_j(noisy_ys[idx], padding_p, pred_ys[idx], kern_idx_j)
          dL_dwj[kern_idx_j] += sum([dL_dy[i] * dy_dw[kern_idx_j][i] for i in range(len(dL_dy))])

      loss_avg /= len(pred_ys)
      loss_history.append(loss_avg)
      dL_dwj = dL_dwj / len(pred_ys)
      w_kernel = w_kernel - learning_rate * dL_dwj

      print(f"Epoch {e} with Loss {loss_avg}")

      # Create a frame visualizing first curve + kernel for this epoch
      idx_plot = 0
      gt = ys[idx_plot]
      pred = pred_ys[idx_plot]
      z_in = noisy_ys[idx_plot]
      save_epoch_frame(xs, gt, pred, z_in, w_kernel, loss_avg, e, frames_dir)

    save_loss_curve(loss_history, epochs, frames_dir)
    create_video(frames_dir, output_dir, run_ts, run_name, fps=10, seconds_per_frame=1)







