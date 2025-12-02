from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from typing import Literal

from grokking.data_utils import DataTracker


def train_val_lines(
    ax: Axes,
    tracker: DataTracker,
    mode: Literal["loss", "acc"],
    model_name: str,
    dataset_name: str,
    y_label: str,
    *,
    x_log: bool = False,
    y_log: bool = False,
    max_y: int | None = None,
) -> None:
    if mode == "loss":
        train_y = np.array(tracker.epoch_train_losses)
        val_y = np.array(tracker.val_losses)
        title_prefix = "Loss per Epoch"
    elif mode == "acc":
        train_y = np.array(tracker.epoch_train_accs) * 100
        val_y = np.array(tracker.val_accs) * 100
        title_prefix = "Accuracy per Epoch"
    else:
        msg = f"Mode must be either 'loss' or 'acc' (got '{mode}')"
        raise ValueError(msg)

    # Avoid zero on x-axis for log scale.
    x = np.arange(len(tracker.epoch_train_losses))
    if x_log or y_log:
        x += 1

    ax.plot(x, train_y, label="Train")
    ax.plot(x, val_y, label="Validation")
    ax.set_title(
        f"{title_prefix} ({model_name} on {dataset_name})",
        size=12,
        pad=14,
        fontweight="bold",
    )
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")
    ax.legend()
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    if max_y is not None:
        margin = (max_y - min(train_y.min(), val_y.min())) * 0.05
        ax.set_ylim(-margin, max_y)


def alt_train_val_lines(
    ax: Axes,
    trackers: list[DataTracker],
    labels: list[str],
    line_colors: list[str],
    mode: Literal["loss", "acc"],
    model_name: str,
    dataset_name: str,
    y_label: str,
    *,
    x_log: bool = False,
    y_log: bool = False,
    legend_loc: str | None = None,
) -> None:
    if mode == "loss":
        train_ys = [np.array(t.epoch_train_losses) for t in trackers]
        val_ys = [np.array(t.val_losses) for t in trackers]
        title_prefix = "Loss per Epoch"
    elif mode == "acc":
        train_ys = [np.array(t.epoch_train_accs) * 100 for t in trackers]
        val_ys = [np.array(t.val_accs) * 100 for t in trackers]
        title_prefix = "Accuracy per Epoch"
    else:
        msg = f"Mode must be either 'loss' or 'acc' (got '{mode}')"
        raise ValueError(msg)

    # Avoid zero on x-axis for log scale.
    x = np.arange(len(train_ys[0]))
    if x_log or y_log:
        x += 1

    for train, val, label, color in zip(
        train_ys, val_ys, labels, line_colors, strict=True
    ):
        ax.plot(x, train, label=f"Train ({label})", linestyle="--", color=color)
        ax.plot(x, val, label=f"Validation ({label})", color=color)

    ax.set_title(
        f"{title_prefix} ({model_name} on {dataset_name})",
        size=12,
        pad=14,
        fontweight="bold",
    )
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")
    ax.legend(loc=legend_loc)
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")


def frac_comparisons(
    figs: tuple[Figure, Figure],
    axes: tuple[Axes, Axes],
    trackers: dict[str, DataTracker],
    bar_ticks: list[float],
    tracker_prefix: str,
    model_name: str,
) -> None:
    def apply_style(ax_: Axes, mode: Literal["loss", "acc"]) -> None:
        title_prefix = "Loss per Epoch" if mode == "loss" else "Accuracy per Epoch"
        ylabel = "Cross-Entropy Loss" if mode == "loss" else "Accuracy (%)"
        ax_.set_title(
            f"{title_prefix} ({model_name} on Division mod 97)",
            size=12,
            pad=14,
            fontweight="bold",
        )
        ax_.set_xlabel("Epoch", fontweight="bold")
        ax_.set_ylabel(ylabel, fontweight="bold")
        ax_.set_xscale("log")
        dashed = Line2D([0], [0], color="gray", linestyle="--", label="Train")
        solid = Line2D([0], [0], color="gray", linestyle="-", label="Validation")
        ax_.legend(handles=[dashed, solid])

    # Create color map for the curves, and ensure correct ordering.
    keys = sorted(k for k in trackers if k.startswith(tracker_prefix))
    base = plt.get_cmap("viridis")
    truncated = mcolors.LinearSegmentedColormap.from_list(
        "viridis_trunc", base(np.linspace(0, 0.95, 256))
    )
    colors = truncated(np.linspace(0, 1, len(keys)))

    # Plot the data using the color map.
    for i, key in enumerate(keys):
        tracker = trackers[key]
        x = np.arange(1, len(tracker.epoch_train_losses) + 1)
        axes[0].plot(
            x,
            tracker.epoch_train_losses,
            label="Train",
            color=colors[i],
            lw=2,
            linestyle="dashed",
        )
        axes[0].plot(x, tracker.val_losses, label="Validation", color=colors[i], lw=2)
        axes[1].plot(
            x,
            np.array(tracker.epoch_train_accs) * 100,
            label="Train",
            color=colors[i],
            lw=2,
            linestyle="dashed",
        )
        axes[1].plot(
            x,
            np.array(tracker.val_accs) * 100,
            label="Validation",
            color=colors[i],
            lw=2,
        )

    # Add a colorbar with the proper color range and labels.
    norm = mcolors.Normalize(vmin=min(bar_ticks), vmax=max(bar_ticks))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=truncated)
    sm.set_array([])
    for fig, ax in zip(figs, axes, strict=True):
        cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.05)
        cbar.set_ticks(bar_ticks)
        cbar.set_label("Train Fraction", fontweight="bold")

    apply_style(axes[0], "loss")
    apply_style(axes[1], "acc")
