"""
tools/analyze_size_distribution.py
Object size distribution analysis for person detection datasets,
following the TinyPerson Benchmark methodology (Yu et al., 2019).

Metrics implemented:
    AS = sqrt(w * h)              — Absolute Size  (section 3, eq. 1)
    RS = sqrt((w*h) / (W*H))      — Relative Size  (section 3, eq. 2)

Usage:
    python tools/analyze_size_distribution.py                              # synthetic data
    python tools/analyze_size_distribution.py --labels L --images I        # real dataset
    python tools/analyze_size_distribution.py --labels L --images I --compare
    python tools/analyze_size_distribution.py --labels L --images I --save
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import lognorm

warnings.filterwarnings("ignore")

# ── Color palette ─────────────────────────────────────────────────────────────
_C = {
    "hist":   "#4C72B0",   # histogram
    "kde":    "#DD8452",   # KDE curve
    "fit":    "#55A868",   # log-normal fit
    "mean":   "#C44E52",   # mean
    "median": "#8172B3",   # median
    "p90":    "#937860",   # 90th percentile
    "tail":   "#E07B54",   # long-tail zone
    "bg":     "#F7F7F7",   # figure background
}

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)


# =============================================================================
# 1. SYNTHETIC DATA GENERATION
#    Simulates an aerial surveillance scenario (similar to VisDrone / Okutama)
#    with varied resolutions and mostly small persons.
# =============================================================================
def create_synthetic_dataset(n_images: int = 800, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic bounding boxes replicating the log-normal distribution
    typical of drone datasets (majority of persons <50 px, long right tail
    from close-range objects).
    """
    rng = np.random.default_rng(seed)

    # Typical resolutions in VisDrone / Okutama-Action
    img_ws = rng.choice([1920, 2048, 1280, 1600, 3840],
                        p=[0.40, 0.25, 0.15, 0.15, 0.05], size=n_images)
    img_hs = rng.choice([1080, 1536,  720, 1200, 2160],
                        p=[0.40, 0.25, 0.15, 0.15, 0.05], size=n_images)

    # Persons per image: negative binomial distribution → right tail
    ppi = np.clip(rng.negative_binomial(n=1, p=0.25, size=n_images) + 1, 1, 60)
    N   = ppi.sum()

    img_w = np.repeat(img_ws, ppi)
    img_h = np.repeat(img_hs, ppi)

    # Bbox heights: log-normal (mu≈33 px, sigma≈0.7) — mostly tiny persons
    bbox_h = np.exp(rng.normal(3.5, 0.7, N)).clip(5, img_h * 0.4).astype(int)
    bbox_w = (bbox_h * rng.uniform(0.30, 0.55, N)).clip(3, img_w * 0.3).astype(int)

    bbox_w = np.minimum(bbox_w, img_w - 1)
    bbox_h = np.minimum(bbox_h, img_h - 1)

    df = pd.DataFrame({
        "image_width":  img_w,
        "image_height": img_h,
        "bbox_width":   bbox_w,
        "bbox_height":  bbox_h,
    })
    print(f"[Synthetic]  {n_images} images  |  {len(df):,} bounding boxes")
    return df


# =============================================================================
# 2. LOADING REAL YOLO LABELS
#    YOLO format: class cx_n cy_n w_n h_n  (normalised values [0,1])
# =============================================================================
def load_yolo_labels(labels_dir: Path, images_dir: Path) -> pd.DataFrame:
    """
    Converts normalised YOLO labels to pixels using each image's actual
    dimensions. Only processes class 0 (person).
    """
    from PIL import Image as PILImage  # local import — not required in synthetic mode

    records = []
    for lf in sorted(labels_dir.glob("*.txt")):
        img_file = next(
            (images_dir / (lf.stem + ext)
             for ext in (".jpg", ".jpeg", ".png", ".bmp")
             if (images_dir / (lf.stem + ext)).exists()),
            None,
        )
        if img_file is None:
            continue

        with PILImage.open(img_file) as img:
            W, H = img.size

        for line in lf.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5 or int(parts[0]) != 0:
                continue
            _, _, _, w_n, h_n = map(float, parts[:5])
            records.append({
                "image_width":  W,
                "image_height": H,
                "bbox_width":   max(1, int(w_n * W)),
                "bbox_height":  max(1, int(h_n * H)),
            })

    df = pd.DataFrame(records)
    print(f"[YOLO real]  {labels_dir.name}  |  {len(df):,} bounding boxes")
    return df


# =============================================================================
# 3. SIZE METRICS — TinyPerson Benchmark (Yu et al., eq. 1 and 2)
# =============================================================================
def compute_sizes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["abs_size"] = np.sqrt(df["bbox_width"]  * df["bbox_height"])
    df["rel_size"] = np.sqrt(
        (df["bbox_width"] * df["bbox_height"]) /
        (df["image_width"] * df["image_height"])
    )
    return df


# =============================================================================
# 4. CONSOLE STATISTICS
# =============================================================================
def print_stats(df: pd.DataFrame, label: str = "") -> None:
    header = f"  SIZE DISTRIBUTION — {label}" if label else "  SIZE DISTRIBUTION"
    print("\n" + "=" * 66)
    print(header)
    print("=" * 66)
    for col, name in [("abs_size", "Absolute Size AS [px]"),
                      ("rel_size", "Relative Size RS [fraction]")]:
        s = df[col]
        tail_flag = "  ← LONG TAIL" if s.skew() > 1.5 else ""
        print(f"\n  {name}")
        print(f"    N              : {len(s):>10,}")
        print(f"    Mean / Median  : {s.mean():>10.4f}  /  {s.median():.4f}")
        print(f"    Std            : {s.std():>10.4f}")
        print(f"    Min / Max      : {s.min():>10.4f}  /  {s.max():.4f}")
        print(f"    p25/p75/p90    : {s.quantile(.25):.4f} / "
              f"{s.quantile(.75):.4f} / {s.quantile(.90):.4f}")
        print(f"    Skewness       : {s.skew():>10.3f}{tail_flag}")
        print(f"    Kurtosis       : {s.kurtosis():>10.3f}")
    print("=" * 66 + "\n")


# =============================================================================
# 5. VISUAL COMPONENTS
# =============================================================================
def _vline(ax: plt.Axes, x: float, label: str, color: str,
           y_frac: float = 0.82, ls: str = "--") -> None:
    """Annotated vertical line with the statistical value."""
    ax.axvline(x, color=color, lw=1.7, ls=ls, zorder=4)
    ylim  = ax.get_ylim()
    y_pos = ylim[0] + (ylim[1] - ylim[0]) * y_frac
    ax.text(
        x * 1.03, y_pos,
        f"{label}\n{x:.3g}",
        color=color, fontsize=8, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.80, ec=color),
    )


def _subplot_distribution(
    df: pd.DataFrame,
    col: str,
    ax_main: plt.Axes,
    ax_ccdf: plt.Axes,
    title: str,
    xlabel: str,
    clip_pct: float = 99.5,
) -> None:
    """
    Fills a pair of axes (distribution + CCDF) for a size metric.

    ax_main: histogram + KDE + log-normal fit
    ax_ccdf: Complementary CDF (log-log) — straight line indicates power law
             (heavy tail); concave curve indicates log-normal.
    """
    data   = df[col].dropna().values
    clip_v = np.percentile(data, clip_pct)
    dv     = data[data <= clip_v]           # clipped data for plotting only

    mean_v   = data.mean()
    median_v = np.median(data)
    p75_v    = np.percentile(data, 75)
    p90_v    = np.percentile(data, 90)
    skew_v   = stats.skew(data)
    kurt_v   = stats.kurtosis(data)

    # ── Histogram ──────────────────────────────────────────────────────────────
    ax_main.hist(dv, bins=60, density=True,
                 color=_C["hist"], alpha=0.30, edgecolor="white",
                 linewidth=0.3, zorder=2, label="Histogram (empirical PDF)")

    # ── KDE ────────────────────────────────────────────────────────────────────
    kde    = stats.gaussian_kde(dv, bw_method="scott")
    x_kde  = np.linspace(dv.min(), dv.max(), 600)
    y_kde  = kde(x_kde)
    ax_main.plot(x_kde, y_kde, color=_C["kde"], lw=2.5, zorder=5,
                 label="Smoothed KDE")
    ax_main.fill_between(x_kde, y_kde, alpha=0.10, color=_C["kde"], zorder=3)

    # ── Log-normal fit ─────────────────────────────────────────────────────────
    shape, loc, scale = lognorm.fit(dv, floc=0)
    x_ln  = np.linspace(dv.min(), dv.max(), 600)
    y_ln  = lognorm.pdf(x_ln, shape, loc=loc, scale=scale)
    ax_main.plot(x_ln, y_ln, color=_C["fit"], lw=2.0, ls="-.",
                 zorder=5, label=f"Log-normal fit (σ={shape:.2f})")

    # ── Long-tail shading (> p75) ──────────────────────────────────────────────
    # The mean−median gap is a direct indicator of skewness:
    # if mean > median, the right tail is pulling the distribution.
    mask = x_kde >= p75_v
    ax_main.fill_between(x_kde[mask], y_kde[mask], alpha=0.22,
                          color=_C["tail"], zorder=3,
                          label=f"Long tail (> p75 = {p75_v:.2g})")

    # ── Statistical lines ──────────────────────────────────────────────────────
    _vline(ax_main, mean_v,   "Mean",   _C["mean"],   0.84)
    _vline(ax_main, median_v, "Median", _C["median"], 0.70, ls=":")
    _vline(ax_main, p90_v,    "p90",    _C["p90"],    0.56, ls=(0, (3, 1, 1, 1)))

    # ── Shape-metrics box ──────────────────────────────────────────────────────
    long_tail_verdict = "Yes" if skew_v > 1 else "No"
    txt = (
        f"Skewness   = {skew_v:.2f}\n"
        f"Kurtosis   = {kurt_v:.2f}\n"
        f"Long tail  : {long_tail_verdict}\n"
        f"N = {len(data):,}"
    )
    ax_main.text(
        0.97, 0.97, txt, transform=ax_main.transAxes,
        fontsize=8.5, va="top", ha="right", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.88, ec="#AAAAAA"),
    )

    # Log X scale when the tail is very long (max/median > 15)
    if data.max() / max(median_v, 1e-9) > 15:
        ax_main.set_xscale("log")
        ax_main.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax_main.set_xlabel(xlabel + "  [log scale]", fontsize=10)
    else:
        ax_main.set_xlabel(xlabel, fontsize=10)

    ax_main.set_ylabel("Probability density", fontsize=10)
    ax_main.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax_main.legend(fontsize=8, loc="upper right", framealpha=0.9, edgecolor="#CCCCCC")
    ax_main.grid(True, alpha=0.35, ls="--")
    ax_main.spines[["top", "right"]].set_visible(False)

    # ── CCDF (Complementary CDF) on log-log scale ─────────────────────────────
    # Sort data and compute exceedance probability P(X > x).
    # A straight line on log-log indicates power law (heavy tail);
    # concave downward curvature indicates log-normal.
    sorted_d  = np.sort(data)
    ccdf      = 1.0 - np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    ax_ccdf.loglog(sorted_d, ccdf + 1e-9,
                   color=_C["kde"], lw=1.8, alpha=0.85, label="Empirical CCDF")

    # Theoretical CCDF from log-normal fit for comparison
    ccdf_ln = 1.0 - lognorm.cdf(sorted_d, shape, loc=loc, scale=scale)
    ax_ccdf.loglog(sorted_d, ccdf_ln + 1e-9, color=_C["fit"],
                   lw=1.5, ls="--", label="Log-normal CCDF")

    ax_ccdf.axvline(p90_v, color=_C["p90"], lw=1.4, ls=":", label=f"p90={p90_v:.2g}")
    ax_ccdf.set_xlabel(xlabel + "  [log]", fontsize=9.5)
    ax_ccdf.set_ylabel("P(X > x)  [log]", fontsize=9.5)
    ax_ccdf.set_title("CCDF — heavy-tail diagnostic", fontsize=10,
                       fontweight="bold", pad=8)
    ax_ccdf.legend(fontsize=7.5, framealpha=0.9, edgecolor="#CCCCCC")
    ax_ccdf.grid(True, alpha=0.3, which="both", ls="--")
    ax_ccdf.spines[["top", "right"]].set_visible(False)


# =============================================================================
# 6. MAIN FIGURE
# =============================================================================
def plot_size_distributions(
    df: pd.DataFrame,
    label: str = "Dataset",
    save_path: Path | None = None,
) -> None:
    """
    2×2 figure:
      Row 0 — AS distribution  (histogram/KDE | CCDF log-log)
      Row 1 — RS distribution  (histogram/KDE | CCDF log-log)
    """
    fig = plt.figure(figsize=(17, 11))
    fig.patch.set_facecolor(_C["bg"])

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.45, wspace=0.32,
        left=0.07, right=0.97, top=0.90, bottom=0.07,
    )

    _subplot_distribution(
        df, "abs_size",
        ax_main=fig.add_subplot(gs[0, 0]),
        ax_ccdf=fig.add_subplot(gs[0, 1]),
        title="Absolute Size (AS)",
        xlabel="AS = √(w · h)   [px]",
    )
    _subplot_distribution(
        df, "rel_size",
        ax_main=fig.add_subplot(gs[1, 0]),
        ax_ccdf=fig.add_subplot(gs[1, 1]),
        title="Relative Size (RS)",
        xlabel="RS = √(w·h / W·H)   [image fraction]",
    )

    fig.suptitle(
        f"Object Size Distribution — {label}\n"
        "Left: PDF (Probability Density Function)  ·  Right: CCDF (heavy-tail diagnostic)",
        fontsize=12, fontweight="bold", y=0.97,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[Figure saved]  → {save_path}")

    plt.show()


# =============================================================================
# 7. COMPARISON PLOT (two datasets side by side)
# =============================================================================
def plot_comparison(
    df_a: pd.DataFrame, label_a: str,
    df_b: pd.DataFrame, label_b: str,
    save_path: Path | None = None,
) -> None:
    """
    Overlays the KDEs of two datasets to directly compare
    their AS and RS distributions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.patch.set_facecolor(_C["bg"])

    colors = [("#4C72B0", "#92B6D5"), ("#DD8452", "#F2C4A0")]

    for ax, col, title, xlabel in zip(
        axes,
        ["abs_size", "rel_size"],
        ["Absolute Size (AS)", "Relative Size (RS)"],
        ["AS = √(w·h)  [px]", "RS = √(w·h/W·H)  [fraction]"],
    ):
        for (df, lbl), (c_line, c_fill) in zip(
            [(df_a, label_a), (df_b, label_b)], colors
        ):
            data  = df[col].dropna().values
            clip  = np.percentile(data, 99.5)
            dv    = data[data <= clip]
            kde   = stats.gaussian_kde(dv, bw_method="scott")
            xg    = np.linspace(dv.min(), dv.max(), 600)
            yg    = kde(xg)
            ax.plot(xg, yg, color=c_line, lw=2.5, label=f"{lbl}  (μ={data.mean():.2g})")
            ax.fill_between(xg, yg, alpha=0.18, color=c_fill)

        if any(df_a[col].max() / max(df_a[col].median(), 1e-9) > 15 for _ in [1]):
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.set_xlabel(xlabel + "  [log scale]", fontsize=10)
        else:
            ax.set_xlabel(xlabel, fontsize=10)

        ax.set_ylabel("Probability density", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.35, ls="--")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Distribution comparison: {label_a} vs {label_b}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[Figure saved]  → {save_path}")

    plt.show()


# =============================================================================
# 8. ENTRY POINT
# =============================================================================
def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels",  type=Path, default=None,
                   help="Path to the YOLO labels folder (.txt). "
                        "If omitted, a synthetic dataset is analyzed instead.")
    p.add_argument("--images",  type=Path, default=None,
                   help="Path to the corresponding images folder. Required with --labels.")
    p.add_argument("--save",    action="store_true",
                   help="Save the figure as PNG in the current directory")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.labels and not args.images:
        print("[ERROR] --images is required when --labels is given")
        raise SystemExit(1)

    if args.labels:
        if not args.labels.exists():
            print(f"[ERROR] Labels folder not found: {args.labels}")
            raise SystemExit(1)
        if not args.images.exists():
            print(f"[ERROR] Images folder not found: {args.images}")
            raise SystemExit(1)

        dataset_name = args.labels.parent.parent.name  # <dataset>/train/labels → <dataset>
        df = compute_sizes(load_yolo_labels(args.labels, args.images))
    else:
        dataset_name = "Synthetic"
        df = compute_sizes(create_synthetic_dataset())

    print_stats(df, dataset_name)
    plot_size_distributions(
        df, label=dataset_name,
        save_path=Path(f"size_distribution_{dataset_name}.png") if args.save else None,
    )

