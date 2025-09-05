import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- data ---
df = pd.DataFrame(
    {
        "model": ["CLEWS", "CQTNet", "ByteCover2", "LIE", "DVINet", "LIE-trans."],
        "preprocessing_mean": [0.52, 0.53, 0.52, 1.67, 0.53, 1.66],
        "inference_avg": [0.66, 1.14, 1.31, 0.22, 1.40, 4.410],
        "total_avg": [1.18, 1.67, 1.83, 1.89, 1.93, 6.07],
        "total_std": [1.11, 2.17, 2.83, 0.513, 2.64, 3.42],
        "params_m": [196.79, 35.18, 202.28, 31.89, 11.47, 31.89],
    }
)

# sort by total time if you want
df = df.sort_values("total_avg").reset_index(drop=True)

y = np.arange(len(df))
pre = df["preprocessing_mean"].to_numpy()
infer = df["inference_avg"].to_numpy()
tot = df["total_avg"].to_numpy()
tstd = df["total_std"].to_numpy()
params = df["params_m"].to_numpy()

# choose a scale so the largest model size fits nicely on the left
right_span = (tot + tstd).max()
left_span = params.max()
scale = right_span / left_span * 0.9  # tweak 0.9 if you want more/less left width
neg_widths = -params * scale  # negative = left bars

fig, ax = plt.subplots(figsize=(8, 4.8))

# stacked runtime (right side)
edge_kwargs = dict(edgecolor="black", linewidth=1.2)
ax.barh(y, pre, color="#bcdffb", **edge_kwargs, label="Preprocessing")
ax.barh(y, infer, left=pre, color="#f8c9a8", **edge_kwargs, label="Inference")

# total std moustache: only to the right
ax.errorbar(
    tot,
    y,
    xerr=[np.zeros_like(tstd), tstd],
    fmt="none",
    ecolor="black",
    elinewidth=1.2,
    capsize=4,
    zorder=3,
)

# model size (left side), with label of true params
bars = ax.barh(
    y, neg_widths, color="#d9d9d9", **edge_kwargs, label="Model size (scaled)"
)
for i, (w, p) in enumerate(zip(neg_widths, params)):
    ax.text(
        w - 0.02 * right_span, y[i], f"{p:.1f}M", va="center", ha="right", fontsize=8
    )

# center divider
ax.axvline(0, color="black", linewidth=1.2)

# labels / ticks
ax.set_yticks(y)
ax.set_yticklabels(df["model"], fontsize=11)
# ax.set_xlabel("Time (s)", fontsize=12)
# ax.tick_params(axis="x", labelsize=11)

# --- nice limits and custom x-ticks/labels (no second axis) ---

# how far right the bars+error go (seconds)
right_span = (tot + tstd).max()

# how big models get (M params)
left_span_m = params.max()

# scaling used for left bars (so size fits visually)
scale = right_span / left_span_m * 0.9  # same scale you used for neg_widths
neg_widths = -params * scale

# choose ‘nice’ tick steps
left_step = 50 if left_span_m > 120 else 10  # 50M or 10M ticks on the left
right_step = 2 if right_span > 5 else 1  # 2s or 1s ticks on the right

# build tick positions (negative for model size, positive for time)
left_ticks_m = np.arange(
    0, np.ceil(left_span_m / left_step) * left_step + 0.1, left_step
)
left_tick_pos = -left_ticks_m * scale
right_ticks_s = np.arange(
    0, np.ceil(right_span / right_step) * right_step + 0.1, right_step
)

# combine (skip the duplicated 0 from the left side)
xticks = np.r_[left_tick_pos[left_tick_pos < 0], right_ticks_s]
xtlabels = [f"{int(m)}M" for m in left_ticks_m[left_tick_pos < 0]] + [
    f"{t:g}" for t in right_ticks_s
]

# apply
ax.set_xticks(xticks)
ax.set_xticklabels(xtlabels)
ax.xaxis.grid(True, linestyle="--", alpha=0.6)

# center divider and margins
ax.axvline(0, color="black", linewidth=1.2)
pad = 0.08 * right_span
ax.set_xlim(neg_widths.min() - pad, right_ticks_s.max() + pad)

# grid on x only
ax.xaxis.grid(True, linestyle="--", alpha=0.6)
ax.yaxis.grid(False)

# limits with some padding
pad = 0.05 * right_span
pad_l = 0.2 * right_span
ax.set_xlim(neg_widths.min() - pad_l, right_span + pad)

# ax.set_ylim(-0.5, len(df) - 0.5)

# legend
ax.legend(frameon=True, ncols=1, fontsize=10, loc="lower right")

plt.tight_layout()
plt.savefig("runtime_params_horizontal.png", dpi=200, bbox_inches="tight")
# plt.show()
