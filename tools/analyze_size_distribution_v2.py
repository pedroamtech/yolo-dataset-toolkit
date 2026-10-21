"""
tools/analyze_size_distribution_v2.py
Size distribution analysis + long-tail size bins and balance plan.

Everything in analyze_size_distribution.py (AS/RS, KDE, log-normal fit, CCDF),
plus the dataset-driven size bins of the Aerial Person Aug pipeline
(size_bins.auto_bins / balance_plan), so the bins and deficits can be inspected
for any YOLO dataset without the pipeline's config module:

  · the long tail starts at the --tail-percentile of AS (default 75 → upper quartile);
  · from there, bins grow geometrically by --bin-ratio (default 1.5) up to the
    largest labeled person; the last bin is open-ended;
  · every tail bin is raised to the --balance-level (default: the count of the most
    populated tail bin, which receives nothing). Persons below the tail start (the
    head) are never augmented.

Differences from v1: AS/RS use float pixel sizes (no int truncation, so counts
match size_bins.py near the bin edges), and images are matched by any common
extension, case-insensitively.

Usage:
    python tools/analyze_size_distribution_v2.py                              # synthetic data
    python tools/analyze_size_distribution_v2.py --labels L --images I
    python tools/analyze_size_distribution_v2.py --labels L --images I --save
    python tools/analyze_size_distribution_v2.py --labels L --images I --bins 25,40,60,90
    python tools/analyze_size_distribution_v2.py --labels L --images I --balance-level 5000
"""

import argparse
import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_size_distribution as v1   # same folder: plots, stats, synthetic data

OPEN_END   = 10_000.0     # upper edge of the last bin (any person fits)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")


# =============================================================================
# 1. LOADING REAL YOLO LABELS (float pixel sizes)
# =============================================================================
def load_yolo_labels(labels_dir: Path, images_dir: Path) -> pd.DataFrame:
    """
    Converts normalised YOLO labels to pixels using each image's actual
    dimensions. Only processes class 0 (person). Sizes stay as floats.
    """
    from PIL import Image as PILImage  # local import — not required in synthetic mode

    images = {p.stem: p for p in images_dir.iterdir()
              if p.suffix.lower() in IMAGE_EXTS}

    records = []
    for lf in sorted(labels_dir.glob("*.txt")):
        img_file = images.get(lf.stem)
        if img_file is None:
            continue

        with PILImage.open(img_file) as img:
            W, H = img.size

        for line in lf.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5 or parts[0] != "0":
                continue
            w_n, h_n = float(parts[3]), float(parts[4])
            records.append({
                "image_width":  W,
                "image_height": H,
                "bbox_width":   w_n * W,
                "bbox_height":  h_n * H,
            })

    df = pd.DataFrame(records)
    print(f"[YOLO real]  {labels_dir.name}  |  {len(df):,} bounding boxes")
    return df


# =============================================================================
# 2. SIZE BINS AND BALANCE PLAN (same rules as size_bins.py)
# =============================================================================
def auto_bins(as_values: np.ndarray, tail_percentile: float = 75,
              ratio: float = 1.5) -> list:
    """Geometric bins from the tail start (percentile of AS) up to the largest person."""
    start = float(np.percentile(as_values, tail_percentile))
    top   = float(as_values.max())
    edges = [start]
    while edges[-1] * ratio < top:
        edges.append(edges[-1] * ratio)
    edges = [round(e, 1) for e in edges] + [OPEN_END]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def parse_bins(text: str) -> list:
    """'25,40,60' → [(25, 40), (40, 60), (60, OPEN_END)]."""
    edges = sorted(float(e) for e in text.split(",") if e.strip())
    if len(edges) < 1:
        raise ValueError("--bins needs at least one edge")
    edges.append(OPEN_END)
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def bin_label(b: tuple) -> str:
    return f"{b[0]:g}-{b[1]:g}" if b[1] < OPEN_END else f"{b[0]:g}-inf"


def balance_plan(as_values: np.ndarray, bins: list, level_rule="dominant_bin",
                 label: str = "", tail_percentile: float = 75,
                 ratio: float = 1.5) -> dict:
    """
    Persons per tail bin, the level every bin should reach and the deficit
    max(0, level − count) per bin. 'dominant_bin' uses the most populated tail
    bin as the level; an integer fixes it.
    """
    counts   = {b: int(((as_values >= b[0]) & (as_values < b[1])).sum()) for b in bins}
    dominant = max(bins, key=lambda b: counts[b])
    level    = counts[dominant] if level_rule == "dominant_bin" else int(level_rule)
    return {
        "dataset":            label,
        "level_rule":         str(level_rule),
        "level":              level,
        "dominant_bin":       bin_label(dominant),
        "tail_percentile":    tail_percentile,
        "bin_ratio":          ratio,
        "persons":            int(len(as_values)),
        "persons_below_tail": int((as_values < bins[0][0]).sum()),
        "percentiles":        {str(q): round(float(np.percentile(as_values, q)), 2)
                               for q in (10, 25, 50, 75, 90, 99)},
        "bins":               [list(b) for b in bins],
        "counts":             {bin_label(b): counts[b] for b in bins},
        "deficit":            {bin_label(b): max(0, level - counts[b]) for b in bins},
    }


def _tail_desc(plan: dict) -> str:
    """'p75 = 25 px AS' for automatic bins, '25 px AS (explicit bins)' otherwise."""
    start = f"{plan['bins'][0][0]:g} px AS"
    if plan["tail_percentile"] is None:
        return f"{start} (explicit bins)"
    return f"p{plan['tail_percentile']:g} = {start}"


def print_plan(plan: dict) -> None:
    print("=" * 66)
    print(f"  BALANCE PLAN — {plan['dataset']}")
    print("=" * 66)
    print(f"  {plan['persons']:,} persons, {plan['persons_below_tail']:,} below the "
          f"tail start (never augmented)")
    ratio = f"   ·   bin ratio ×{plan['bin_ratio']:g}" if plan["bin_ratio"] else ""
    print(f"  Tail start   : {_tail_desc(plan)}{ratio}")
    print(f"  Level        : {plan['level']:,} ({plan['level_rule']}; "
          f"dominant tail bin {plan['dominant_bin']})")
    print(f"\n  {'AS bin [px]':>16} {'persons':>9} {'deficit':>9}")
    for k in plan["counts"]:
        tag = "  (no augmentation)" if plan["deficit"][k] == 0 else ""
        print(f"  {k:>16} {plan['counts'][k]:9,d} {plan['deficit'][k]:9,d}{tag}")
    total = sum(plan["deficit"].values())
    print(f"  {'total':>16} {sum(plan['counts'].values()):9,d} {total:9,d}")
    print("=" * 66 + "\n")


# =============================================================================
# 3. FIGURE
# =============================================================================
def _plot_bin_edges(ax: plt.Axes, bins: list) -> None:
    """Thin vertical lines at every bin edge on the AS distribution panel."""
    for i, (lo, _) in enumerate(bins):
        ax.axvline(lo, color="#555555", lw=0.8, ls="-", alpha=0.55, zorder=3,
                   label="Size-bin edges" if i == 0 else None)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9, edgecolor="#CCCCCC")


def _plot_balance(ax: plt.Axes, plan: dict) -> None:
    """Persons per tail bin, with the deficit stacked on top up to the level."""
    keys    = list(plan["counts"])
    counts  = np.array([plan["counts"][k] for k in keys])
    deficit = np.array([plan["deficit"][k] for k in keys])
    x       = np.arange(len(keys))

    ax.bar(x, counts, width=0.7, color=v1._C["hist"], edgecolor="white",
           linewidth=2, zorder=3, label="Labeled persons")
    ax.bar(x, deficit, width=0.7, bottom=counts, color="white",
           edgecolor=v1._C["tail"], hatch="///", linewidth=1.2, zorder=3,
           label="Deficit (to augment)")
    ax.axhline(plan["level"], color=v1._C["mean"], lw=1.5, ls="--", zorder=4,
               label=f"Level = {plan['level']:,} ({plan['level_rule']})")

    for xi, c, d in zip(x, counts, deficit):
        ax.text(xi, c + d, f"+{d:,}" if d else f"{c:,}", ha="center", va="bottom",
                fontsize=8, color="#333333")

    # Headroom above the level for the value labels and the legend
    ax.set_ylim(0, max(plan["level"], counts.max()) * 1.35)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=8.5)
    ax.set_xlabel("AS bin  [px]", fontsize=10)
    ax.set_ylabel("Persons", fontsize=10)
    ax.set_title(
        f"Long-tail size bins — tail starts at {_tail_desc(plan)}  ·  "
        f"{plan['persons_below_tail']:,} persons below "
        f"(head, never augmented)",
        fontsize=11, fontweight="bold", pad=10)
    ax.legend(fontsize=8, loc="upper center", ncol=3, framealpha=0.9, edgecolor="#CCCCCC")
    ax.grid(True, axis="y", alpha=0.35, ls="--")
    ax.grid(False, axis="x")
    ax.spines[["top", "right"]].set_visible(False)


def plot_size_distributions(df: pd.DataFrame, plan: dict, label: str = "Dataset",
                            save_path: Path | None = None) -> None:
    """
    3-row figure:
      Row 0 — AS distribution with bin edges  (histogram/KDE | CCDF log-log)
      Row 1 — RS distribution                 (histogram/KDE | CCDF log-log)
      Row 2 — persons and deficit per tail bin
    """
    fig = plt.figure(figsize=(17, 16))
    fig.patch.set_facecolor(v1._C["bg"])

    gs = gridspec.GridSpec(
        3, 2, figure=fig, height_ratios=[1, 1, 0.9],
        hspace=0.50, wspace=0.32,
        left=0.07, right=0.97, top=0.92, bottom=0.05,
    )

    ax_as = fig.add_subplot(gs[0, 0])
    v1._subplot_distribution(
        df, "abs_size", ax_main=ax_as, ax_ccdf=fig.add_subplot(gs[0, 1]),
        title="Absolute Size (AS)", xlabel="AS = √(w · h)   [px]",
    )
    _plot_bin_edges(ax_as, [tuple(b) for b in plan["bins"]])

    v1._subplot_distribution(
        df, "rel_size", ax_main=fig.add_subplot(gs[1, 0]),
        ax_ccdf=fig.add_subplot(gs[1, 1]),
        title="Relative Size (RS)", xlabel="RS = √(w·h / W·H)   [image fraction]",
    )

    _plot_balance(fig.add_subplot(gs[2, :]), plan)

    fig.suptitle(
        f"Object Size Distribution — {label}\n"
        "PDF · CCDF (heavy-tail diagnostic) · long-tail balance plan",
        fontsize=12, fontweight="bold", y=0.98,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[Figure saved]  → {save_path}")

    plt.show()


# =============================================================================
# 4. ENTRY POINT
# =============================================================================
def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels",  type=Path, default=None,
                   help="Path to the YOLO labels folder (.txt). "
                        "If omitted, a synthetic dataset is analyzed instead.")
    p.add_argument("--images",  type=Path, default=None,
                   help="Path to the corresponding images folder. Required with --labels.")
    p.add_argument("--tail-percentile", type=float, default=75,
                   help="AS percentile where the long tail starts (default 75)")
    p.add_argument("--bin-ratio", type=float, default=1.5,
                   help="Geometric growth of the tail bins (default 1.5)")
    p.add_argument("--bins", type=str, default=None,
                   help="Explicit AS bin edges in px, e.g. '25,40,60,90' "
                        "(the last bin is open-ended). Overrides the automatic bins.")
    p.add_argument("--balance-level", type=str, default="dominant_bin",
                   help="Count every tail bin should reach: 'dominant_bin' (default) "
                        "or an integer")
    p.add_argument("--save",    action="store_true",
                   help="Save the figure (PNG) and the balance plan (JSON) "
                        "in the current directory")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.labels and not args.images:
        print("[ERROR] --images is required when --labels is given")
        raise SystemExit(1)
    if args.bin_ratio <= 1:
        print("[ERROR] --bin-ratio must be greater than 1")
        raise SystemExit(1)
    if args.balance_level != "dominant_bin" and not args.balance_level.isdigit():
        print("[ERROR] --balance-level must be 'dominant_bin' or an integer")
        raise SystemExit(1)

    if args.labels:
        if not args.labels.exists():
            print(f"[ERROR] Labels folder not found: {args.labels}")
            raise SystemExit(1)
        if not args.images.exists():
            print(f"[ERROR] Images folder not found: {args.images}")
            raise SystemExit(1)

        dataset_name = args.labels.parent.parent.name  # <dataset>/train/labels → <dataset>
        df = load_yolo_labels(args.labels, args.images)
    else:
        dataset_name = "Synthetic"
        df = v1.create_synthetic_dataset()

    if len(df) < 50:
        print(f"[ERROR] Only {len(df)} labeled persons — too few for a size analysis.")
        raise SystemExit(1)

    df = v1.compute_sizes(df)
    as_values = df["abs_size"].to_numpy(dtype=float)
    try:
        bins = parse_bins(args.bins) if args.bins else \
            auto_bins(as_values, args.tail_percentile, args.bin_ratio)
    except ValueError as e:
        print(f"[ERROR] Invalid --bins: {e}")
        raise SystemExit(1)

    plan = balance_plan(as_values, bins, args.balance_level, dataset_name,
                        None if args.bins else args.tail_percentile,
                        None if args.bins else args.bin_ratio)

    v1.print_stats(df, dataset_name)
    print_plan(plan)

    if args.save:
        json_path = Path(f"balance_plan_{dataset_name}.json")
        json_path.write_text(json.dumps(plan, indent=2))
        print(f"[Plan saved]    → {json_path}")

    plot_size_distributions(
        df, plan, label=dataset_name,
        save_path=Path(f"size_distribution_v2_{dataset_name}.png") if args.save else None,
    )
