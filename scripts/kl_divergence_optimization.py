"""
kl_divergence_optimization.py

Side-by-side animation of two KL-divergence minimisation runs that both start
from the same Q₀ (a single Gaussian at μ=0.5, σ=0.15):

  Left  panel: minimise KL(P ‖ Q)  — forward KL  (mean/moment-matching)
  Right panel: minimise KL(Q ‖ P)  — reverse KL  (mode-seeking)

P  = bimodal Gaussian PMF with equal-height peaks at x=0.25 and x=0.75,
     discretised over a 500-bucket grid on [0, 1].
Q  = single Gaussian PMF parameterised by (μ, σ), optimised via Adam +
     numerical gradients.
"""

import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import petname

# ---------------------------------------------------------------------------
# Paths / aesthetics
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

BG_DARK    = "#1E1E1E"
BG_MID     = "#2A2A2A"
TEXT_COLOR = "#DDDDDD"
WHITE      = "white"
COLOR_P    = "#FFFFFF"   # white  – ground truth P
COLOR_Q    = "#E87B4C"   # orange – hypothesis Q

N_BUCKETS = 500
X = np.linspace(0.0, 1.0, N_BUCKETS)
_EPS = 1e-12

# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

def make_P() -> np.ndarray:
    """Bimodal Gaussian PMF: equal-height peaks at x=0.25 and x=0.75."""
    sigma_p = 0.07
    p = (np.exp(-0.5 * ((X - 0.25) / sigma_p) ** 2) +
         np.exp(-0.5 * ((X - 0.75) / sigma_p) ** 2))
    p /= p.sum()
    return p


def make_Q(mu: float, sigma: float) -> np.ndarray:
    """Single Gaussian PMF on the shared grid, clipped and renormalised."""
    q = np.exp(-0.5 * ((X - mu) / sigma) ** 2)
    q = np.clip(q, _EPS, None)
    q /= q.sum()
    return q


# ---------------------------------------------------------------------------
# KL divergences
# ---------------------------------------------------------------------------

def kl_forward(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P ‖ Q) = Σ P · log(P / Q)"""
    mask = p > _EPS
    return float(np.sum(p[mask] * np.log(p[mask] / np.clip(q[mask], _EPS, None))))


def kl_reverse(q: np.ndarray, p: np.ndarray) -> float:
    """KL(Q ‖ P) = Σ Q · log(Q / P)"""
    mask = q > _EPS
    return float(np.sum(q[mask] * np.log(q[mask] / np.clip(p[mask], _EPS, None))))


# ---------------------------------------------------------------------------
# Adam optimiser
# ---------------------------------------------------------------------------

class Adam:
    def __init__(self, lr: float = 1e-2, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.m     = np.zeros(2)
        self.v     = np.zeros(2)
        self.t     = 0

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2
        m_hat  = self.m / (1 - self.beta1 ** self.t)
        v_hat  = self.v / (1 - self.beta2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# Numerical gradient (central differences in (μ, log σ) space)
# ---------------------------------------------------------------------------

def numerical_grad(loss_fn, mu: float, log_sigma: float,
                   h: float = 1e-5) -> tuple[float, float]:
    d_mu = (loss_fn(mu + h, log_sigma) - loss_fn(mu - h, log_sigma)) / (2 * h)
    d_ls = (loss_fn(mu, log_sigma + h) - loss_fn(mu, log_sigma - h)) / (2 * h)
    return d_mu, d_ls


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

def run_optimization(
    P: np.ndarray,
    direction: str,      # 'forward'  →  KL(P‖Q),  'reverse'  →  KL(Q‖P)
    n_steps: int = 300,
    lr: float   = 6e-3,
    mu0: float  = 0.5,
    sigma0: float = 0.15,
) -> list[tuple[float, float, float]]:
    """Return trajectory: [(μ, σ, kl_value), …] for each gradient step."""

    if direction == 'forward':
        loss_fn = lambda m, ls: kl_forward(P, make_Q(m, np.exp(ls)))
    else:
        loss_fn = lambda m, ls: kl_reverse(make_Q(m, np.exp(ls)), P)

    mu        = mu0
    log_sigma = np.log(sigma0)
    opt       = Adam(lr=lr)
    trajectory: list[tuple[float, float, float]] = []

    for _ in range(n_steps):
        val = loss_fn(mu, log_sigma)
        trajectory.append((mu, np.exp(log_sigma), val))

        g_mu, g_ls = numerical_grad(loss_fn, mu, log_sigma)
        params     = opt.step(np.array([mu, log_sigma]), np.array([g_mu, g_ls]))
        mu         = float(np.clip(params[0], 0.01, 0.99))
        log_sigma  = float(np.clip(params[1], np.log(0.005), np.log(0.49)))

    return trajectory


# ---------------------------------------------------------------------------
# Frame renderer
# ---------------------------------------------------------------------------

def make_frame(
    P: np.ndarray,
    traj_fwd: list,
    traj_rev: list,
    step: int,
    n_steps: int,
    out_path: str,
) -> None:
    mu_f, sig_f, kl_f = traj_fwd[step]
    mu_r, sig_r, kl_r = traj_rev[step]
    Q_fwd = make_Q(mu_f, sig_f)
    Q_rev = make_Q(mu_r, sig_r)

    y_max = max(P.max(), Q_fwd.max(), Q_rev.max()) * 1.3

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(16, 7),
        facecolor=BG_DARK,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.10, wspace=0.18)
    fig.suptitle(
        "KL Divergence Minimisation",
        fontsize=14, fontweight="bold", color=WHITE, y=0.97,
    )

    def draw_panel(ax, Q, kl_val, title_line1, metric_str):
        ax.set_facecolor(BG_MID)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)

        ax.fill_between(X, P, alpha=0.45, color=COLOR_P)
        ax.fill_between(X, Q, alpha=0.55, color=COLOR_Q)
        ax.plot(X, P, color=COLOR_P, lw=1.2, alpha=0.9, label="P  (Ground Truth)")
        ax.plot(X, Q, color=COLOR_Q, lw=2.0,             label="Q  (Hypothesis)")

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, y_max)
        ax.set_xlabel("x", color=TEXT_COLOR, fontsize=11)
        ax.set_ylabel("Probability Mass", color=TEXT_COLOR, fontsize=11)

        # Title: two lines — optimisation name, then current metric value
        ax.set_title(
            f"{title_line1}\n{metric_str} = {kl_val:.5f} nats",
            color=WHITE, fontsize=13, pad=10,
        )
        ax.legend(
            fontsize=10, facecolor=BG_MID,
            edgecolor="#555555", labelcolor=TEXT_COLOR,
            loc="upper left",
        )

    draw_panel(ax_l, Q_fwd, kl_f, "Minimise  KL(P ‖ Q)", "KL(P ‖ Q)")
    draw_panel(ax_r, Q_rev, kl_r, "Minimise  KL(Q ‖ P)", "KL(Q ‖ P)")

    fig.text(
        0.5, 0.02,
        f"[KL(P‖Q)] Q Gaussian Parameters:  μ = {mu_f:.4f}   σ = {sig_f:.4f}"
        f"          Step {step + 1} / {n_steps}          "
        f"[KL(Q‖P)] Q Gaussian Parameters:  μ = {mu_r:.4f}   σ = {sig_r:.4f}",
        ha="center", va="bottom", fontsize=9, color=TEXT_COLOR,
    )

    plt.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    stamp   = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    name    = petname.generate(2, "_")
    run_tag = f"{stamp}_{name}"

    tmp_dir = OUTPUT_DIR / f"{run_tag}_kl_divergence_frames"
    tmp_dir.mkdir(exist_ok=True)

    P = make_P()

    # Both runs share the same Q₀
    MU0    = 0.5
    SIGMA0 = 0.15
    N_STEPS = 300

    print("Running forward KL(P ‖ Q) optimisation …")
    traj_fwd = run_optimization(P, 'forward', n_steps=N_STEPS, lr=6e-3,
                                mu0=MU0, sigma0=SIGMA0)

    print("Running reverse KL(Q ‖ P) optimisation …")
    traj_rev = run_optimization(P, 'reverse', n_steps=N_STEPS, lr=6e-3,
                                mu0=MU0, sigma0=SIGMA0)

    kl_f_final = traj_fwd[-1][2]
    kl_r_final = traj_rev[-1][2]
    mu_f_final, sig_f_final, _ = traj_fwd[-1]
    mu_r_final, sig_r_final, _ = traj_rev[-1]
    print(f"  Forward KL converged → μ={mu_f_final:.4f}  σ={sig_f_final:.4f}  "
          f"KL(P‖Q)={kl_f_final:.5f}")
    print(f"  Reverse KL converged → μ={mu_r_final:.4f}  σ={sig_r_final:.4f}  "
          f"KL(Q‖P)={kl_r_final:.5f}")

    print(f"\nRendering {N_STEPS} frames …")
    frame_paths = []
    for step in range(N_STEPS):
        out_path = tmp_dir / f"frame_{step:04d}.png"
        make_frame(P, traj_fwd, traj_rev, step, N_STEPS, str(out_path))
        frame_paths.append(out_path)
        if (step + 1) % 50 == 0:
            print(f"  {step + 1} / {N_STEPS}")

    # Encode with ffmpeg using the image-sequence input pattern
    FPS = 30
    out_video = OUTPUT_DIR / f"{run_tag}_kl_divergence.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", str(tmp_dir / "frame_%04d.png"),
        "-vf", "scale=1920:-2:flags=lanczos",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        str(out_video),
    ]
    print("\nEncoding video …")
    subprocess.run(cmd, check=True)
    print(f"\nDone!  Video saved to:\n  {out_video}")

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
