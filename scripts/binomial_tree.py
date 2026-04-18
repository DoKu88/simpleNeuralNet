import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── style ─────────────────────────────────────────────────────────────────────
BG          = "#111111"
EDGE_COL    = "#AAAAAA"
WHITE_NODE  = "#EEEEEE"
GREY_NODE   = "#555555"
WHITE_TEXT  = "#111111"
GREY_TEXT   = "#EEEEEE"
BORDER_COL  = "#CCCCCC"

# ── layout ────────────────────────────────────────────────────────────────────
N_MAX      = 5
X_SPACING  = 2.2
Y_SPACING  = 2.0
BOX_W      = 1.6
BOX_H      = 1.4

def pos(n, k):
    x = (k - n / 2) * X_SPACING
    y = (N_MAX - n) * Y_SPACING
    return x, y

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 11))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# ── edges ─────────────────────────────────────────────────────────────────────
for n in range(N_MAX):
    for k in range(n + 1):
        x1, y1 = pos(n, k)
        for dk in (0, 1):
            x2, y2 = pos(n + 1, k + dk)
            ax.plot(
                [x1, x2], [y1 - BOX_H / 2, y2 + BOX_H / 2],
                color=EDGE_COL, linewidth=1.2, zorder=1
            )

# ── nodes ─────────────────────────────────────────────────────────────────────
for n in range(N_MAX + 1):
    for k in range(n + 1):
        x, y = pos(n, k)
        white = k <= 2
        face  = WHITE_NODE if white else GREY_NODE
        text  = WHITE_TEXT if white else GREY_TEXT

        rect = mpatches.FancyBboxPatch(
            (x - BOX_W / 2, y - BOX_H / 2),
            BOX_W, BOX_H,
            boxstyle="round,pad=0.06",
            facecolor=face,
            edgecolor=BORDER_COL,
            linewidth=1.4,
            zorder=2,
        )
        ax.add_patch(rect)

        ax.text(x, y, rf"$\binom{{{n}}}{{{k}}}$",
                ha="center", va="center", fontsize=28,
                color=text, zorder=3)

# ── axes ──────────────────────────────────────────────────────────────────────
pad_x = BOX_W
pad_y = BOX_H + 0.3
x_vals = [(k - N_MAX / 2) * X_SPACING for k in range(N_MAX + 1)]
ax.set_xlim(min(x_vals) - pad_x, max(x_vals) + pad_x)
ax.set_ylim(-pad_y, N_MAX * Y_SPACING + pad_y)
ax.set_aspect("equal")
ax.axis("off")


plt.tight_layout()
plt.savefig("outputs/binomial_tree.png", dpi=150, bbox_inches="tight",
            facecolor=BG)
print("Saved to outputs/binomial_tree.png")
plt.close()
