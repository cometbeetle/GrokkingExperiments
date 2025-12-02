from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Lambda
from torch.utils.data import random_split
import matplotlib.pyplot as plt

from grokking import set_seed
from grokking.models import (
    GrokTransformer,
    GrokMLP,
    DeepGrokMLPMSE,
    DeepGrokMLPCE,
    GrokEmbedMLP,
    GrokResNetCE,
    GrokResNetMSE,
)
from grokking.data_utils import (
    ModOpDataset,
    DataTracker,
    save_checkpoint,
    load_checkpoint,
)
from grokking.plot_utils import train_val_lines, alt_train_val_lines, frac_comparisons


DEVICE = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device("cpu")
)

TRANSFORMER_TRAIN_FRACS: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5]

EMBED_MLP_TRAIN_FRACS: list[float] = [0.3, 0.4, 0.5, 0.6, 0.7]


def run_transformer_mod(ckpt_dir: Path | None, device: torch.device) -> None:
    dataset = ModOpDataset(97, "/", squeeze_labels=True)

    # Run transformer experiments.
    for frac in TRANSFORMER_TRAIN_FRACS:
        model = GrokTransformer(97).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
        tracker = DataTracker()

        model.run_train(
            1000,
            128,
            dataset,
            optimizer,
            train_frac=frac,
            data_tracker=tracker,
        )

        if ckpt_dir is not None:
            save_checkpoint(
                model,
                optimizer,
                tracker,
                ckpt_dir / f"mod_transformer_{int(frac * 100)}.pth",
            )


def run_mlp_mod(ckpt_dir: Path | None, device: torch.device) -> None:
    dataset = ModOpDataset(97, "/", squeeze_labels=True)

    # Run embedding experiments.
    for frac in EMBED_MLP_TRAIN_FRACS:
        model = GrokEmbedMLP(97).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
        tracker = DataTracker()

        model.run_train(
            1000,
            128,
            dataset,
            optimizer,
            train_frac=frac,
            data_tracker=tracker,
        )

        if ckpt_dir is not None:
            save_checkpoint(
                model,
                optimizer,
                tracker,
                ckpt_dir / f"mod_embed_mlp_{int(frac * 100)}.pth",
            )

    # Run non-embedding experiments.
    model = GrokMLP().to(device)
    tracker = DataTracker()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    dataset.squeeze_labels = False
    model.run_train(1000, 128, dataset, optimizer, train_frac=0.5, data_tracker=tracker)
    if ckpt_dir is not None:
        save_checkpoint(model, optimizer, tracker, ckpt_dir / "mod_standard_mlp.pth")


def run_mlp_mnist(ckpt_dir: Path | None, device: torch.device) -> None:
    dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=Compose([ToTensor(), Lambda(lambda x: torch.flatten(x))]),
    )

    # Use only 10% of the MNIST training data.
    dataset, _ = random_split(dataset, [0.1, 0.9])

    # Run MLP model experiments.
    deep_model = DeepGrokMLPMSE().to(device)
    deep_model_no_scale = DeepGrokMLPMSE(weight_scale=1.0).to(device)
    deep_model_ce = DeepGrokMLPCE().to(device)
    shallow_model = DeepGrokMLPMSE(n_layers=3, weight_scale=8.0).to(device)
    shallow_model_no_scale = DeepGrokMLPMSE(n_layers=3, weight_scale=1.0).to(device)
    shallow_model_ce = DeepGrokMLPCE(n_layers=3, weight_scale=8.0).to(device)
    experiments = [
        (deep_model, "mnist_deep_mlp.pth"),
        (deep_model_no_scale, "mnist_deep_mlp_no_scale.pth"),
        (deep_model_ce, "mnist_deep_mlp_ce.pth"),
        (shallow_model, "mnist_shallow_mlp.pth"),
        (shallow_model_no_scale, "mnist_shallow_mlp_no_scale.pth"),
        (shallow_model_ce, "mnist_shallow_mlp_ce.pth"),
    ]

    for model, checkpoint_name in experiments:
        tracker = DataTracker()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.001)
        model.run_train(
            2000, 128, dataset, optimizer, train_frac=0.25, data_tracker=tracker
        )
        if ckpt_dir is not None:
            save_checkpoint(model, optimizer, tracker, ckpt_dir / checkpoint_name)


def run_resnet(ckpt_dir: Path | None, device: torch.device) -> None:
    dataset = datasets.SVHN(root="data", download=True, transform=ToTensor())

    # Use only 10% of the SVHN training data.
    dataset, _ = random_split(dataset, [0.1, 0.9])

    # Run ResNet-18 experiments.
    model_ce = GrokResNetCE().to(device)
    model_no_scale = GrokResNetCE(weight_scale=1.0).to(device)
    model_mse = GrokResNetMSE(weight_scale=120.0).to(device)
    experiments = [
        (model_ce, "svhn_resnet.pth"),
        (model_no_scale, "svhn_resnet_no_scale.pth"),
        (model_mse, "svhn_resnet_mse.pth"),
    ]
    for model, checkpoint_name in experiments:
        tracker = DataTracker()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
        model.run_train(
            1000,
            128,
            dataset,
            optimizer,
            train_frac=0.7,
            data_tracker=tracker,
        )
        if ckpt_dir is not None:
            save_checkpoint(model, optimizer, tracker, ckpt_dir / checkpoint_name)


def make_plots(plot_dir: Path, ckpt_dir: Path) -> None:
    trackers: dict[str, DataTracker] = {}

    for file in ckpt_dir.iterdir():
        if not file.is_file() or file.suffix != ".pth":
            continue

        # Load checkpoint data trackers.
        trackers[file.stem] = load_checkpoint(file)["data_tracker"]

    # -----------------------------------------------------------
    # ------------------------ RESNET-18 ------------------------
    # -----------------------------------------------------------

    fig, ax = plt.subplots()
    train_val_lines(
        ax,
        trackers["svhn_resnet"],
        "loss",
        "ResNet-18",
        "SVHN",
        "Cross-Entropy Loss",
        x_log=True,
        y_log=True,
    )
    fig.savefig(plot_dir / "svhn_resnet_loss.svg", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    train_val_lines(
        ax,
        trackers["svhn_resnet"],
        "acc",
        "ResNet-18",
        "SVHN",
        "Accuracy (%)",
        x_log=True,
    )
    fig.savefig(plot_dir / "svhn_resnet_acc.svg", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    alt_train_val_lines(
        ax,
        [trackers["svhn_resnet_no_scale"], trackers["svhn_resnet_mse"]],
        ["no scaling", "MSE loss"],
        ["tab:green", "tab:red"],
        "loss",
        "ResNet-18",
        "SVHN",
        "Loss",
        x_log=True,
        y_log=True,
    )
    fig.savefig(plot_dir / "svhn_resnet_loss_others.svg", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    alt_train_val_lines(
        ax,
        [trackers["svhn_resnet_no_scale"], trackers["svhn_resnet_mse"]],
        ["no scaling", "MSE loss"],
        ["tab:green", "tab:red"],
        "acc",
        "ResNet-18",
        "SVHN",
        "Accuracy (%)",
        x_log=True,
        legend_loc="upper left",
    )
    fig.savefig(plot_dir / "svhn_resnet_acc_others.svg", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------
    # -------------------------- MNIST --------------------------
    # -----------------------------------------------------------

    fig, ax = plt.subplots()
    train_val_lines(
        ax,
        trackers["mnist_deep_mlp"],
        "loss",
        "12-Layer MLP",
        "MNIST",
        "Mean Squared Error",
        x_log=True,
        y_log=True,
    )
    fig.savefig(plot_dir / "mnist_deep_mlp_loss.svg", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    train_val_lines(
        ax,
        trackers["mnist_shallow_mlp"],
        "loss",
        "3-Layer MLP",
        "MNIST",
        "Mean Squared Error",
        x_log=True,
        y_log=True,
    )
    fig.savefig(plot_dir / "mnist_shallow_mlp_loss.svg", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    train_val_lines(
        ax,
        trackers["mnist_deep_mlp"],
        "acc",
        "12-Layer MLP",
        "MNIST",
        "Accuracy (%)",
        x_log=True,
    )
    fig.savefig(plot_dir / "mnist_deep_mlp_acc.svg", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    train_val_lines(
        ax,
        trackers["mnist_shallow_mlp"],
        "acc",
        "3-Layer MLP",
        "MNIST",
        "Accuracy (%)",
        x_log=True,
    )
    fig.savefig(plot_dir / "mnist_shallow_mlp_acc.svg", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    alt_train_val_lines(
        ax,
        [trackers["mnist_deep_mlp_no_scale"], trackers["mnist_deep_mlp_ce"]],
        ["no scaling", "CE loss"],
        ["tab:green", "tab:red"],
        "loss",
        "12-Layer MLP",
        "MNIST",
        "Loss",
        x_log=True,
        y_log=True,
    )
    fig.savefig(
        plot_dir / "mnist_deep_mlp_loss_others.svg", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    fig, ax = plt.subplots()
    alt_train_val_lines(
        ax,
        [trackers["mnist_deep_mlp_no_scale"], trackers["mnist_deep_mlp_ce"]],
        ["no scaling", "CE loss"],
        ["tab:green", "tab:red"],
        "acc",
        "12-Layer MLP",
        "MNIST",
        "Accuracy (%)",
        x_log=True,
    )
    fig.savefig(
        plot_dir / "mnist_deep_mlp_acc_others.svg", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    fig, ax = plt.subplots()
    alt_train_val_lines(
        ax,
        [trackers["mnist_shallow_mlp_no_scale"], trackers["mnist_shallow_mlp_ce"]],
        ["no scaling", "CE loss"],
        ["tab:green", "tab:red"],
        "loss",
        "3-Layer MLP",
        "MNIST",
        "Loss",
        x_log=True,
        y_log=True,
    )
    fig.savefig(
        plot_dir / "mnist_shallow_mlp_loss_others.svg", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    fig, ax = plt.subplots()
    alt_train_val_lines(
        ax,
        [trackers["mnist_shallow_mlp_no_scale"], trackers["mnist_shallow_mlp_ce"]],
        ["no scaling", "CE loss"],
        ["tab:green", "tab:red"],
        "acc",
        "3-Layer MLP",
        "MNIST",
        "Accuracy (%)",
        x_log=True,
    )
    fig.savefig(
        plot_dir / "mnist_shallow_mlp_acc_others.svg", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # -------------------- MODULAR ARITHMETIC --------------------
    # ------------------------------------------------------------

    # TRANSFORMER

    fig_loss, ax_loss = plt.subplots()
    fig_acc, ax_acc = plt.subplots()
    frac_comparisons(
        (fig_loss, fig_acc),
        (ax_loss, ax_acc),
        trackers,
        TRANSFORMER_TRAIN_FRACS,
        "mod_transformer_",
        "Transformer",
    )
    fig_loss.savefig(
        plot_dir / "mod_transformer_loss.svg", dpi=300, bbox_inches="tight"
    )
    fig_acc.savefig(plot_dir / "mod_transformer_acc.svg", dpi=300, bbox_inches="tight")
    plt.close(fig_loss)
    plt.close(fig_acc)

    # EMBEDDED MULTI-LAYER PERCEPTRON

    fig_loss, ax_loss = plt.subplots()
    fig_acc, ax_acc = plt.subplots()
    frac_comparisons(
        (fig_loss, fig_acc),
        (ax_loss, ax_acc),
        trackers,
        EMBED_MLP_TRAIN_FRACS,
        "mod_embed_mlp_",
        "EmbedMLP",
    )
    fig_loss.savefig(plot_dir / "mod_embed_mlp_loss.svg", dpi=300, bbox_inches="tight")
    fig_acc.savefig(plot_dir / "mod_embed_mlp_acc.svg", dpi=300, bbox_inches="tight")
    plt.close(fig_loss)
    plt.close(fig_acc)

    # STANDARD MULTI-LAYER PERCEPTRON

    fig, ax = plt.subplots()
    train_val_lines(
        ax,
        trackers["mod_standard_mlp"],
        "loss",
        "Standard MLP",
        "Division mod 97",
        "Mean Squared Error",
        x_log=True,
        y_log=True,
    )
    fig.savefig(plot_dir / "mod_standard_mlp_loss.svg", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    train_val_lines(
        ax,
        trackers["mod_standard_mlp"],
        "acc",
        "Standard MLP",
        "Division mod 97",
        "Accuracy (%)",
        x_log=True,
    )
    fig.savefig(plot_dir / "mod_standard_mlp_acc.svg", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    set_seed(246)

    parser = argparse.ArgumentParser(
        description="Run the grokking experiments, and / or create the figures."
    )
    parser.add_argument(
        "-t", "--run-train", action="store_true", help="run the training loops"
    )
    parser.add_argument(
        "-p", "--make-plots", action="store_true", help="generate plots from the data"
    )
    parser.add_argument(
        "-c",
        "--ckpt-dir",
        type=Path,
        default=Path("checkpoints"),
        help="directory used to save / load the model checkpoints (default: 'checkpoints')",
    )
    parser.add_argument(
        "-s",
        "--save-models",
        type=bool,
        default=True,
        help="whether to save the trained models (default: True)",
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        default=DEVICE,
        help="device to use (defaults to GPU)",
    )
    args = parser.parse_args()

    # Create the checkpoint storage directory if it does not exist.
    if args.save_models and not args.ckpt_dir.exists():
        args.ckpt_dir.mkdir(parents=True)

    # Create the plotting directory if it does not exist.
    plot_dir = Path("plots")
    if args.make_plots and not plot_dir.exists():
        plot_dir.mkdir(parents=True)

    # Run the training experiments, and save their results to checkpoint files.
    if args.run_train:
        ckpt_dir = args.ckpt_dir if args.save_models else None
        run_transformer_mod(ckpt_dir, args.device)
        run_mlp_mod(ckpt_dir, args.device)
        run_mlp_mnist(ckpt_dir, args.device)
        run_resnet(ckpt_dir, args.device)

    # Create the plots.
    if args.make_plots:
        make_plots(plot_dir, args.ckpt_dir)


if __name__ == "__main__":
    main()
