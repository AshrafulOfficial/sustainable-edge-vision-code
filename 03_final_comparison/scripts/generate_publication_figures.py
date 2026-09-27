"""
============================================================
Publication Figure Generator
Project:
Sustainable Edge Vision:
Energy-Efficient Deep Learning for Intelligent Crop Disease Detection

Author:
Md. Ashraful Islam

Description:
Generate publication-quality figures for thesis and journal paper.

Outputs:
PNG (600 DPI)
PDF
SVG
============================================================
"""

from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "02_after_pruning_quantization" / "results"

OUTPUT_DIR = PROJECT_ROOT / "03_final_comparison" / "publication_figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PRUNING = RESULTS_DIR / "pruning_5_6_7_8_9_10_11_final_comparison.csv"

print("=" * 60)
print("Publication Figure Generator")
print("=" * 60)

# ============================================================
# Publication Style
# ============================================================

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 17,
    "axes.labelsize": 14,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linestyle": "--",
    "legend.frameon": True,
})

# ============================================================
# Color Palette
# ============================================================

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "black": "#222222"
}

# ============================================================
# Helper Functions
# ============================================================

def save_figure(fig, filename):
    """
    Save figure in PNG, PDF and SVG formats.
    """
    for ext in ["png", "pdf"]:
        fig.savefig(
            OUTPUT_DIR / f"{filename}.{ext}",
            dpi=600,
            bbox_inches="tight"
        )

    print(f"[OK] Saved: {filename}")


def load_pruning_csv():
    """
    Load and preprocess pruning comparison CSV.
    """

    if not CSV_PRUNING.exists():
        raise FileNotFoundError(CSV_PRUNING)

    df = pd.read_csv(CSV_PRUNING)

    # Remove duplicate pruning ratios
    df = (
        df
        .drop_duplicates(subset=["pruning_ratio"])
        .sort_values("pruning_ratio")
        .reset_index(drop=True)
    )

    # Convert fraction → percentage
    df["pruning_ratio"] *= 100

    accuracy_columns = [
        "best_val_accuracy",
        "baseline_val_accuracy",
    ]

    for col in accuracy_columns:
        if col in df.columns:
            df[col] *= 100

    reduction_columns = [
        "parameter_reduction_percent",
        "mac_reduction_percent",
    ]

    for col in reduction_columns:
        if col in df.columns:
            if df[col].max() <= 1:
                df[col] *= 100

    print("\nLoaded CSV Successfully")
    print(df[[
        "pruning_ratio",
        "best_val_accuracy",
        "parameter_reduction_percent",
        "mac_reduction_percent"
    ]])

    return df


# ============================================================
# Figure 01
# Pruning Ratio vs Validation Accuracy
# ============================================================

def generate_figure_01(df):

    print("\nGenerating Figure 01...")

    x = df["pruning_ratio"]
    y = df["best_val_accuracy"]

    baseline = df["baseline_val_accuracy"].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Main Curve
    ax.plot(
        x,
        y,
        color=COLORS["blue"],
        linewidth=3,
        marker="o",
        markersize=9,
        markerfacecolor="white",
        markeredgewidth=2,
        label="Pruned MobileNetV2",
    )

    # Baseline Reference
    ax.axhline(
        baseline,
        color=COLORS["red"],
        linestyle="--",
        linewidth=2,
        label=f"Baseline ({baseline:.2f}%)",
    )

    # Highlight Final Selected Model (11%)
    final_row = df[df["pruning_ratio"] == 11]

    if not final_row.empty:

        fx = final_row["pruning_ratio"].values[0]
        fy = final_row["best_val_accuracy"].values[0]

        ax.scatter(
            fx,
            fy,
            s=250,
            color=COLORS["green"],
            edgecolor="black",
            linewidth=1.5,
            zorder=5,
            label="Selected Final Model (11%)",
        )

        ax.annotate(
            "Selected Final Model",
            xy=(fx, fy),
            xytext=(9.2, fy - 0.35),
            arrowprops=dict(arrowstyle="->", lw=1.5),
            fontsize=11,
        )

    # Value Labels
    for xx, yy in zip(x, y):

        ax.text(
            xx,
            yy + 0.05,
            f"{yy:.2f}",
            ha="center",
            fontsize=9,
        )

    ax.set_xlabel("Structured Pruning Ratio (%)", fontweight="bold")
    ax.set_ylabel("Validation Accuracy (%)", fontweight="bold")

    ax.set_title(
        "Validation Accuracy under Different Structured Pruning Ratios",
        fontweight="bold",
    )

    ax.set_xticks(x)

    # Zoom without misleading
    ymin = min(min(y), baseline) - 0.5
    ymax = max(max(y), baseline) + 0.3

    ax.set_ylim(ymin, ymax)

    ax.legend()

    ax.grid(True, alpha=0.5)

    plt.tight_layout()

    save_figure(
        fig,
        "Figure_01_Pruning_vs_Validation_Accuracy",
    )

    plt.close()

    print("Figure 01 Completed.")

# ============================================================
# Figure 02
# Structured Pruning Ratio vs Parameter Reduction
# ============================================================

def generate_figure_02(df):

    print("\nGenerating Figure 02...")

    x = df["pruning_ratio"]
    y = df["parameter_reduction_percent"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Main Curve
    ax.plot(
        x,
        y,
        color=COLORS["green"],
        linewidth=3,
        marker="o",
        markersize=9,
        markerfacecolor="white",
        markeredgewidth=2,
        label="Parameter Reduction",
    )

    # Highlight Final Model
    final_row = df[df["pruning_ratio"] == 11]

    if not final_row.empty:

        fx = final_row["pruning_ratio"].values[0]
        fy = final_row["parameter_reduction_percent"].values[0]

        ax.scatter(
            fx,
            fy,
            s=320,
            marker="*",
            color="gold",
            edgecolor="black",
            linewidth=1.3,
            zorder=6,
            label="Final Model"
        )

        ax.annotate(
            f"Final Model\n{fy:.2f}% Reduction",
            xy=(fx, fy),
            xytext=(8.7,19),
            arrowprops=dict(
                arrowstyle="->",
                lw=1.2
            ),
            fontsize=10
        )

    # Value Labels
    for xx, yy in zip(x, y):

        ax.text(
            xx,
            yy+0.35,
            f"{yy:.2f}",
            ha="center",
            fontsize=8,
        )

    ax.set_xlabel(
        "Structured Pruning Ratio (%)",
        fontweight="bold"
    )

    ax.set_ylabel(
        "Parameter Reduction (%)",
        fontweight="bold"
    )

    ax.set_title(
        "Effect of Structured Pruning on Parameter Reduction",
        fontweight="bold",
    )   

    ax.set_xticks(x)
    ax.set_ylim(9, 22)

    ax.set_ylim(
        y.min()-2,
        y.max()+2
    )

    ax.legend()

    ax.grid(True, alpha=0.5)

    plt.tight_layout()

    save_figure(
        fig,
        "Figure_02_Pruning_vs_Parameter_Reduction"
    )

    plt.close()

    print("Figure 02 Completed.")

# ============================================================
# Figure 03
# Dual-Axis Accuracy vs MAC Reduction
# ============================================================

def generate_figure_03(df):

    print("\nGenerating Figure 03...")

    x = df["pruning_ratio"]

    accuracy = df["best_val_accuracy"]
    mac = df["mac_reduction_percent"]

    fig, ax1 = plt.subplots(figsize=(10,6))

    # --------------------------------------------------
    # Accuracy (Left Axis)
    # --------------------------------------------------

    ax1.plot(
        x,
        accuracy,
        color="#1f77b4",
        marker="o",
        linewidth=3,
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=2,
        label="Validation Accuracy"
    )

    ax1.set_xlabel(
        "Structured Pruning Ratio (%)",
        fontweight="bold"
    )

    ax1.set_ylabel(
        "Validation Accuracy (%)",
        color="#1f77b4",
        fontweight="bold"
    )

    ax1.tick_params(
        axis="y",
        labelcolor="#1f77b4"
    )

    ax1.set_xticks(x)

    ax1.set_ylim(
        accuracy.min()-0.4,
        accuracy.max()+0.4
    )

    # --------------------------------------------------
    # MAC Reduction (Right Axis)
    # --------------------------------------------------

    ax2 = ax1.twinx()

    ax2.plot(
        x,
        mac,
        color="#2ca02c",
        marker="s",
        linewidth=3,
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=2,
        label="MAC Reduction"
    )

    ax2.set_ylabel(
        "MAC Reduction (%)",
        color="#2ca02c",
        fontweight="bold"
    )

    ax2.tick_params(
        axis="y",
        labelcolor="#2ca02c"
    )

    ax2.set_ylim(
        mac.min()-2,
        mac.max()+2
    )

    # --------------------------------------------------
    # Highlight Selected Model
    # --------------------------------------------------

    row = df[df["pruning_ratio"] == 11]

    if not row.empty:

        xx = row["pruning_ratio"].values[0]
        yy1 = row["best_val_accuracy"].values[0]
        yy2 = row["mac_reduction_percent"].values[0]

        ax1.scatter(
            xx,
            yy1,
            s=220,
            color="red",
            edgecolor="black",
            zorder=10
        )

        ax2.scatter(
            xx,
            yy2,
            s=220,
            color="gold",
            edgecolor="black",
            zorder=10
        )

        ax2.annotate(
        "20.16% MAC Reduction",
        xy=(xx, yy2),
        xytext=(8.4, 19.5),
        fontsize=10,
        color="#2ca02c",
        fontweight="bold",
        arrowprops=dict(
            arrowstyle="->",
            lw=1.5,
            color="#2ca02c"
            )
        )

        ax1.annotate(
            "Selected Final Model",
            xy=(xx,yy1),
            xytext=(9,yy1-0.35),
            fontsize=10,
            arrowprops=dict(
                arrowstyle="->",
                lw=1.5
            )
        )

    # --------------------------------------------------
    # Legends
    # --------------------------------------------------

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1+lines2,
        labels1+labels2,
        loc="upper center"
    )

    plt.title(
        "Accuracy–Efficiency Trade-off under Structured Pruning",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    save_figure(
        fig,
        "Figure_03_Accuracy_vs_MAC_Reduction"
    )

    plt.close()

    print("Figure 03 Completed.")

# ============================================================
# Figure 04
# Pareto Frontier
# Accuracy vs Parameter Reduction
# ============================================================

def generate_figure_04(df):

    print("\nGenerating Figure 04...")

    x = df["parameter_reduction_percent"]
    y = df["best_val_accuracy"]

    fig, ax = plt.subplots(figsize=(9,6))

    # ----------------------------------------------------
    # Scatter Points
    # ----------------------------------------------------

    ax.scatter(
        x,
        y,
        s=90,
        color="#2E86DE",
        edgecolor="black",
        zorder=3,
        label="Pruned Models"
    )

    # ----------------------------------------------------
    # Connect models
    # ----------------------------------------------------

    ax.plot(
        x,
        y,
        color="gray",
        linestyle="--",
        alpha=.6
    )

    # ----------------------------------------------------
    # Baseline
    # ----------------------------------------------------

    baseline_acc = df["baseline_val_accuracy"].iloc[0]

    ax.scatter(
        0,
        baseline_acc,
        s=180,
        marker="^",
        color="#E74C3C",
        edgecolor="black",
        label="Baseline"
    )

    ax.annotate(
        "Baseline",
        xy=(0, baseline_acc),
        xytext=(2.5, baseline_acc+0),
        fontsize=10,
        arrowprops=dict(arrowstyle="->")
    )

    # ----------------------------------------------------
    # Final Model
    # ----------------------------------------------------

    final = df[df["pruning_ratio"]==11]

    fx = final["parameter_reduction_percent"].values[0]
    fy = final["best_val_accuracy"].values[0]

    ax.scatter(
        fx,
        fy,
        s=300,
        marker="*",
        color="gold",
        edgecolor="black",
        linewidth=1.2,
        zorder=5,
        label="Final Selected Model"
    )

    ax.annotate(
        "Best Practical Trade-off",
        xy=(fx,fy),
        xytext=(13.5,99.35),
        fontsize=11,
        color="green",
        arrowprops=dict(
            arrowstyle="->",
            color="green",
            lw=2
        )
    )

    # ----------------------------------------------------
    # Labels
    # ----------------------------------------------------

    for xx,yy,p in zip(
        x,
        y,
        df["pruning_ratio"]
    ):

        ax.text(
            xx,
            yy+0.03,
            f"{int(p)}%",
            fontsize=9,
            ha="center"
        )

    ax.set_xlabel(
        "Parameter Reduction (%)",
        fontweight="bold"
    )

    ax.set_ylabel(
        "Validation Accuracy (%)",
        fontweight="bold"
    )

    ax.set_title(
        "Pareto Frontier of Accuracy vs Model Compression",
        fontweight="bold"
    )

    ax.grid(alpha=.3)

    ax.legend()

    plt.tight_layout()

    save_figure(
        fig,
        "Figure_04_Pareto_Frontier"
    )

    plt.close()

    print("Figure 04 Completed.")

# ============================================================
# Figure 05
# Impact of Safe Pointwise Incremental Pruning
# ============================================================

def generate_figure_05(df):

    print("\nGenerating Figure 05...")

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = [
        "Parameter\nReduction",
        "MAC\nReduction",
        "Model Size\nReduction",
        "Latency\nReduction",
        "Energy\nReduction"
    ]

    structured = np.array([
        20.57,
        20.16,
        72.72,
        26.21,
        27.78
    ])

    incremental = np.array([
        23.53,
        21.86,
        73.84,
        26.39,
        28.43
    ])

    improvement = incremental - structured

    y = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(11,7))

    bar_height = 0.36

        # --------------------------------------------------------
    # Draw Bars
    # --------------------------------------------------------

    bars1 = ax.barh(
        y - bar_height/2,
        structured,
        height=bar_height,
        color="#2F80ED",
        edgecolor="black",
        linewidth=1.0,
        label="11% Structured Pruning",
        zorder=3
    )

    bars2 = ax.barh(
        y + bar_height/2,
        incremental,
        height=bar_height,
        color="#27AE60",
        edgecolor="black",
        linewidth=1.0,
        label="+2% Incremental Pruning",
        zorder=3
    )

    # --------------------------------------------------------
    # Value Labels
    # --------------------------------------------------------

    for bar in bars1:

        width = bar.get_width()

        ax.text(
            width + 0.8,
            bar.get_y() + bar.get_height()/2,
            f"{width:.2f}%",
            va="center",
            fontsize=10,
            fontweight="bold"
        )

    for bar in bars2:

        width = bar.get_width()

        ax.text(
            width + 0.8,
            bar.get_y() + bar.get_height()/2,
            f"{width:.2f}%",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#1B5E20"
        )

    # --------------------------------------------------------
    # Axis
    # --------------------------------------------------------

    ax.set_yticks(y)
    ax.set_yticklabels(
        metrics,
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Reduction (%)",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_title(
        "Impact of Safe Pointwise Incremental Pruning on Model Efficiency",
        fontsize=17,
        fontweight="bold",
        pad=15
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.30
    )

    ax.set_axisbelow(True)

    ax.legend(
        loc="upper right",
        fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=False
    )

    ax.set_xlim(0, 85)

        # --------------------------------------------------------
    # Improvement Labels
    # --------------------------------------------------------

    for i in range(len(metrics)):

        x1 = structured[i]
        x2 = incremental[i]

        gain = improvement[i]

        ax.annotate(
            "",
            xy=(x2, y[i] + bar_height/2),
            xytext=(x1, y[i] - bar_height/2),
            arrowprops=dict(
                arrowstyle="->",
                color="#D35400",
                lw=2
            )
        )

        ax.text(
            max(x1, x2) + 3,
            y[i],
            f"+{gain:.2f}%",
            color="#D35400",
            fontsize=11,
            fontweight="bold",
            va="center"
        )

    # --------------------------------------------------------
    # Highlight Box
    # --------------------------------------------------------

    summary = (
        "Incremental Pruning Benefits\n"
        "• Parameters: +2.96%\n"
        "• MACs: +1.70%\n"
        "• Model Size: +1.12%\n"
        "• Latency: +0.18%\n"
        "• Energy: +0.65%"
    )

    ax.text(
        0.98,
        0.03,
        summary,
        transform=ax.transAxes,
        fontsize=10,
        ha="right",
        va="bottom",
        bbox=dict(
            facecolor="#FFF8DC",
            edgecolor="gray",
            boxstyle="round,pad=0.6",
            alpha=0.95
        )
    )

    # --------------------------------------------------------
    # Bottom Note
    # --------------------------------------------------------

    fig.text(
        0.5,
        0.02,
        "Safe pointwise incremental pruning consistently improved computational efficiency over the selected 11% structured-pruned MobileNetV2.",
        ha="center",
        fontsize=10,
        style="italic"
    )

    plt.tight_layout(rect=[0,0.05,1,1])

    save_figure(
        fig,
        "Figure_05_Incremental_Pruning_Improvement"
    )

    plt.close()

    print("Figure 05 Completed.")

# ============================================================
# Figure 06
# Overall Performance Comparison Dashboard
# ============================================================

def generate_figure_06(df):

    print("\nGenerating Figure 06...")

    # --------------------------------------------------------
    # Final Models
    # --------------------------------------------------------

    models = [
        "Baseline",
        "11% Structured\nPruning",
        "Incremental\nPruning"
    ]

    # --------------------------------------------------------
    # Final Metrics
    # --------------------------------------------------------

    metrics = {

        "Accuracy (%)": [
            99.69,
            98.38,
            99.43
        ],

        "Parameter Reduction (%)": [
            0.00,
            20.57,
            23.53
        ],

        "MAC Reduction (%)": [
            0.00,
            20.16,
            21.86
        ],

        "Model Size Reduction (%)": [
            0.00,
            72.72,
            73.84
        ],

        "Latency Reduction (%)": [
            0.00,
            26.21,
            26.39
        ],

        "Energy Reduction (%)": [
            0.00,
            27.78,
            28.43
        ]

    }

    # --------------------------------------------------------
    # Colors
    # --------------------------------------------------------

    model_colors = [
        "#8E8E93",     # Baseline (Gray)
        "#2F80ED",     # Structured (Blue)
        "#27AE60"      # Incremental (Green)
    ]

    # --------------------------------------------------------
    # Figure Layout
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18,10)
    )

    axes = axes.flatten()

    metric_names = list(metrics.keys())

    # --------------------------------------------------------
    # Plot Loop
    # --------------------------------------------------------

    for idx, metric in enumerate(metric_names):

        ax = axes[idx]

        values = metrics[metric]

        bars = ax.bar(
            models,
            values,
            color=model_colors,
            edgecolor="black",
            linewidth=1.2,
            width=0.60,
            zorder=3
        )

        # Save for next part
        ax._bars = bars
        ax._values = values
        ax._metric = metric

                # --------------------------------------------------------
        # Value Labels
        # --------------------------------------------------------

        for bar in bars:

            height = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + max(values)*0.02,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

        # --------------------------------------------------------
        # Highlight Best Model
        # --------------------------------------------------------

        best_index = int(np.argmax(values))

        bars[best_index].set_edgecolor("#D62728")
        bars[best_index].set_linewidth(3)

        ax.text(
            bars[best_index].get_x() + bars[best_index].get_width()/2,
            values[best_index] + max(values)*0.10,
            "★ BEST",
            color="#D62728",
            fontsize=11,
            ha="center",
            fontweight="bold"
        )

        # --------------------------------------------------------
        # Axis Formatting
        # --------------------------------------------------------

        ax.set_title(
            metric,
            fontsize=14,
            fontweight="bold",
            pad=12
        )

        ymax = max(values)

        if ymax < 40:
            ax.set_ylim(0, ymax + 8)

        elif ymax < 80:
            ax.set_ylim(0, ymax + 12)

        else:
            ax.set_ylim(0, 105)

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.30
        )

        ax.set_axisbelow(True)

        ax.tick_params(
            axis="x",
            labelsize=10
        )

        ax.tick_params(
            axis="y",
            labelsize=10
        )

        # --------------------------------------------------------
        # Clean Publication Style
        # --------------------------------------------------------

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

            # ========================================================
    # Overall Figure Title
    # ========================================================

    fig.suptitle(
        "Overall Performance Comparison of Baseline, Structured-Pruned and Incrementally Pruned Models",
        fontsize=20,
        fontweight="bold",
        y=0.98
    )

    # ========================================================
    # Custom Legend
    # ========================================================

    from matplotlib.patches import Patch

    legend_elements = [

        Patch(
            facecolor="#8E8E93",
            edgecolor="black",
            label="Baseline"
        ),

        Patch(
            facecolor="#2F80ED",
            edgecolor="black",
            label="11% Structured Pruning"
        ),

        Patch(
            facecolor="#27AE60",
            edgecolor="black",
            label="11% + 2% Incremental Pruning"
        )

    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=3,
        frameon=True,
        fontsize=11,
        bbox_to_anchor=(0.5,0.935)
    )

    # ========================================================
    # Bottom Caption
    # ========================================================

    fig.text(
        0.5,
        0.025,
        "The incremental pruning strategy consistently improved computational efficiency "
        "while preserving high classification accuracy across all evaluation metrics.",
        ha="center",
        fontsize=11,
        style="italic"
    )

    # ========================================================
    # Adjust Layout
    # ========================================================

    plt.subplots_adjust(

        top=0.86,
        bottom=0.10,
        left=0.06,
        right=0.98,
        hspace=0.40,
        wspace=0.28

    )

    # ========================================================
    # Save Figure
    # ========================================================

    save_figure(

        fig,
        "Figure_06_Overall_Model_Comparison"

    )

    plt.close()

    print("Figure 06 Completed.")

# ============================================================
# Figure 07
# Trade-off Analysis Dashboard
# ============================================================

def generate_figure_07(df):

    print("\nGenerating Figure 07...")

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = [
        "Baseline",
        "11% Structured\nPruning",
        "Incremental\nPruning"
    ]

    # --------------------------------------------------------
    # Performance Data
    # --------------------------------------------------------

    accuracy = np.array([
        99.69,
        98.38,
        99.43
    ])

    energy = np.array([
        0.00,
        27.78,
        28.43
    ])

    latency = np.array([
        0.00,
        26.21,
        26.39
    ])

    model_size = np.array([
        0.00,
        72.72,
        73.84
    ])

    # Estimated CO₂ reduction
    # Replace later with actual CodeCarbon values
    co2 = np.array([
        0.00,
        27.78,
        28.43
    ])

    # Relative per-image energy
    image_energy = np.array([
        100.00,
        72.22,
        71.57
    ])

    # --------------------------------------------------------
    # Colors
    # --------------------------------------------------------

    model_colors = [
        "#8E8E93",
        "#2F80ED",
        "#27AE60"
    ]

    # --------------------------------------------------------
    # Dashboard Layout
    # --------------------------------------------------------

    fig, axes = plt.subplots(

        3,
        2,

        figsize=(18,14)

    )

    fig.suptitle(

        "Trade-off Analysis",

        fontsize=22,

        fontweight="bold",

        y=0.985

    )

    # ========================================================
    # Plot Information
    # ========================================================

    plot_info = [

        (
            energy,
            "Accuracy vs Energy Reduction",
            "Energy Reduction (%)",
            accuracy,
            "Validation Accuracy (%)"
        ),

        (
            latency,
            "Accuracy vs Latency Reduction",
            "Latency Reduction (%)",
            accuracy,
            "Validation Accuracy (%)"
        ),

        (
            model_size,
            "Accuracy vs Model Size Reduction",
            "Model Size Reduction (%)",
            accuracy,
            "Validation Accuracy (%)"
        ),

        (
            co2,
            "Accuracy vs CO₂ Reduction",
            "Estimated CO₂ Reduction (%)",
            accuracy,
            "Validation Accuracy (%)"
        )

    ]

        # ========================================================
    # First Four Trade-off Plots
    # ========================================================

    for idx, (x, title, xlabel, y, ylabel) in enumerate(plot_info):

        ax = axes.flat[idx]

        # ----------------------------------------------------
        # Optimization Path
        # ----------------------------------------------------

        ax.plot(
            x,
            y,
            color="#7F8C8D",
            linewidth=2,
            linestyle="--",
            zorder=1
        )

        # ----------------------------------------------------
        # Scatter Points
        # ----------------------------------------------------

        for i in range(len(models)):

            ax.scatter(
                x[i],
                y[i],
                s=180,
                color=model_colors[i],
                edgecolor="black",
                linewidth=1.2,
                zorder=3
            )

        # ----------------------------------------------------
        # Point Labels
        # ----------------------------------------------------

        for i in range(len(models)):

            ax.annotate(
                models[i].replace("\n", " "),
                (x[i], y[i]),
                xytext=(6, 8),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold"
            )

        # ----------------------------------------------------
        # Highlight Final Model
        # ----------------------------------------------------

        ax.scatter(
            x[-1],
            y[-1],
            s=350,
            facecolors="none",
            edgecolors="#D62728",
            linewidth=2.5,
            zorder=4
        )

        # ----------------------------------------------------
        # Arrow showing optimization direction
        # ----------------------------------------------------

        ax.annotate(
            "",
            xy=(x[2], y[2]),
            xytext=(x[1], y[1]),
            arrowprops=dict(
                arrowstyle="->",
                lw=2,
                color="#444444"
            )
        )

        # ----------------------------------------------------
        # Titles & Labels
        # ----------------------------------------------------

        ax.set_title(
            title,
            fontsize=13,
            fontweight="bold"
        )

        ax.set_xlabel(
            xlabel,
            fontsize=11,
            fontweight="bold"
        )

        ax.set_ylabel(
            ylabel,
            fontsize=11,
            fontweight="bold"
        )

        # ----------------------------------------------------
        # Grid
        # ----------------------------------------------------

        ax.grid(
            linestyle="--",
            alpha=0.30
        )

        ax.set_axisbelow(True)

        # ----------------------------------------------------
        # Clean Style
        # ----------------------------------------------------

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_linewidth(1.1)
        ax.spines["bottom"].set_linewidth(1.1)

        # ----------------------------------------------------
        # Dynamic Limits
        # ----------------------------------------------------

        ax.set_xlim(min(x) - 2, max(x) + 5)
        ax.set_ylim(min(y) - 1.0, max(y) + 0.6)

            # ========================================================
    # Per-image Energy Comparison
    # ========================================================

    ax = axes[2, 0]

    bars = ax.bar(

        models,

        image_energy,

        color=model_colors,

        edgecolor="black",

        linewidth=1.2,

        width=0.60,

        zorder=3

    )

    # Value Labels

    for bar in bars:

        height = bar.get_height()

        ax.text(

            bar.get_x() + bar.get_width()/2,

            height + 2,

            f"{height:.2f}%",

            ha="center",

            fontsize=10,

            fontweight="bold"

        )

    # Highlight Final Model

    bars[-1].set_edgecolor("#D62728")
    bars[-1].set_linewidth(2.8)

    ax.set_title(

        "Relative Per-image Energy Consumption",

        fontsize=13,

        fontweight="bold"

    )

    ax.set_ylabel(

        "Relative Energy (%)",

        fontsize=11,

        fontweight="bold"

    )

    ax.grid(

        axis="y",

        linestyle="--",

        alpha=0.30

    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ========================================================
    # Summary Panel
    # ========================================================

    ax = axes[2,1]

    ax.axis("off")

    summary_text = (
        "KEY FINDINGS\n"
        "────────────────────────────\n\n"

        "• Structured pruning reduced\n"
        "  parameters and MACs while\n"
        "  preserving high accuracy.\n\n"

        "• Incremental pruning further\n"
        "  improved efficiency with\n"
        "  minimal accuracy loss.\n\n"

        "• Model size decreased by\n"
        "  more than 73%.\n\n"

        "• Energy consumption reduced\n"
        "  by approximately 28%.\n\n"

        "• The final optimized model\n"
        "  provides the best trade-off\n"
        "  between accuracy and edge\n"
        "  deployment efficiency."
    )

    ax.text(

        0.02,

        0.98,

        summary_text,

        transform=ax.transAxes,

        fontsize=11,

        va="top",

        bbox=dict(

            boxstyle="round",

            facecolor="#F8F9FA",

            edgecolor="#888888",

            linewidth=1.5

        )

    )

    # ========================================================
    # Figure Legend
    # ========================================================

    from matplotlib.lines import Line2D

    legend_items = [

        Line2D(

            [0],

            [0],

            marker="o",

            color="w",

            markerfacecolor=model_colors[0],

            markeredgecolor="black",

            markersize=10,

            label="Baseline"

        ),

        Line2D(

            [0],

            [0],

            marker="o",

            color="w",

            markerfacecolor=model_colors[1],

            markeredgecolor="black",

            markersize=10,

            label="11% Structured Pruning"

        ),

        Line2D(

            [0],

            [0],

            marker="o",

            color="w",

            markerfacecolor=model_colors[2],

            markeredgecolor="black",

            markersize=10,

            label="Incremental Pruning"

        )

    ]

    fig.legend(

        handles=legend_items,

        loc="upper center",

        ncol=3,

        fontsize=11,

        frameon=True,

        bbox_to_anchor=(0.5,0.955)

    )

        # ========================================================
    # Overall Caption
    # ========================================================

    fig.text(

        0.5,

        0.015,

        "Comprehensive trade-off analysis of the baseline, structured-pruned, and "
        "incrementally pruned MobileNetV2 models. The proposed incremental pruning "
        "strategy consistently improves computational efficiency while maintaining "
        "high validation accuracy, making it suitable for sustainable edge deployment.",

        ha="center",

        fontsize=10,

        style="italic"

    )

    # ========================================================
    # Layout Adjustment
    # ========================================================

    plt.subplots_adjust(

        top=0.90,

        bottom=0.08,

        left=0.06,

        right=0.98,

        hspace=0.42,

        wspace=0.30

    )

    # ========================================================
    # Save Figure
    # ========================================================

    save_figure(

        fig,

        "Figure_07_Tradeoff_Analysis"

    )

    plt.close()

    print("Figure 07 Completed.")

# ============================================================
# Figure 08
# Multi-objective Bubble Plot
# ============================================================

def generate_figure_08(df):

    print("\nGenerating Figure 08...")

    # ========================================================
    # Models
    # ========================================================

    models = [

        "Baseline",

        "11% Structured",

        "Incremental"

    ]

    # ========================================================
    # Data
    # ========================================================

    accuracy = np.array([

        99.69,

        98.38,

        99.43

    ])

    mac = np.array([

        0.00,

        20.16,

        20.90

    ])

    energy = np.array([

        0.00,

        27.78,

        28.43

    ])

    model_size = np.array([

        0.00,

        72.72,

        73.84

    ])

    # ========================================================
    # Bubble Size Scaling
    # ========================================================

    bubble_size = np.interp(

        model_size,

        (model_size.min(), model_size.max()),

        (450,1800)

    )

    # ========================================================
    # Figure
    # ========================================================

    fig, ax = plt.subplots(

        figsize=(11,8)

    )

    # ========================================================
    # Bubble Plot
    # ========================================================

    scatter = ax.scatter(

        mac,

        accuracy,

        s=bubble_size,

        c=energy,

        cmap="viridis",

        alpha=0.82,

        edgecolors="black",

        linewidth=1.6,

        zorder=3

    )

    # ========================================================
    # Optimization Path
    # ========================================================

    ax.plot(

        mac,

        accuracy,

        linestyle="--",

        linewidth=2,

        color="#666666",

        zorder=2

    )

    # ========================================================
    # Highlight Final Model
    # ========================================================

    ax.scatter(

        mac[-1],

        accuracy[-1],

        s=bubble_size[-1]*1.12,

        facecolors="none",

        edgecolors="red",

        linewidth=3,

        zorder=5

    )

        # ========================================================
    # Model Annotations
    # ========================================================

    offsets = [

        (-2.0, 0.12),   # Baseline
        (-3.2,-0.35),   # Structured
        (0.55, 0.12)    # Incremental

    ]

    for i, model in enumerate(models):

        dx, dy = offsets[i]

        ax.text(

            mac[i] + dx,

            accuracy[i] + dy,

            f"{model}\nAcc: {accuracy[i]:.2f}%",

            fontsize=10,

            fontweight="bold",

            bbox=dict(

                facecolor="white",

                edgecolor="gray",

                boxstyle="round,pad=0.35",

                alpha=0.95

            )

        )

    # ========================================================
    # Selected Final Model Annotation
    # ========================================================

    ax.annotate(

        "Selected Final Model",

        xy=(mac[-1], accuracy[-1]),

        xytext=(12.5,99.75),

        fontsize=11,

        fontweight="bold",

        color="#D62728",

        arrowprops=dict(

            arrowstyle="->",

            color="#D62728",

            lw=2

        )

    )

    # ========================================================
    # Bubble Size Legend
    # ========================================================

    legend_sizes = [25, 50, 75]

    legend_handles = []

    for size in legend_sizes:

        legend_handles.append(

            plt.scatter(

                [],

                [],

                s=np.interp(size,[0,75],[450,1800]),

                facecolor="lightgray",

                edgecolor="black",

                alpha=0.7,

                label=f"{size:.0f}% Size Reduction"

            )

        )

    legend1 = ax.legend(

        handles=legend_handles,

        title="Bubble Size",

        loc="upper left",

        bbox_to_anchor=(0.12, 0.98),

        fontsize=9,

        frameon=True

    )

    ax.add_artist(legend1)

    # ========================================================
    # Colorbar
    # ========================================================

    cbar = plt.colorbar(

        scatter,

        ax=ax,

        pad=0.02

    )

    cbar.set_label(

        "Energy Reduction (%)",

        fontsize=11,

        fontweight="bold"

    )

    # ========================================================
    # Axis Labels
    # ========================================================

    ax.set_xlabel(

        "MAC Reduction (%)",

        fontsize=13,

        fontweight="bold"

    )

    ax.set_ylabel(

        "Validation Accuracy (%)",

        fontsize=13,

        fontweight="bold"

    )

    ax.set_title(

        "Multi-objective Performance Trade-off",

        fontsize=18,

        fontweight="bold"

    )

        # ========================================================
    # Optimization Direction
    # ========================================================

    ax.annotate(

        "",

        xy=(mac[1], accuracy[1]),

        xytext=(mac[0], accuracy[0]),

        arrowprops=dict(

            arrowstyle="->",

            lw=2,

            color="#555555"

        )

    )

    ax.annotate(

        "",

        xy=(mac[2], accuracy[2]),

        xytext=(mac[1], accuracy[1]),

        arrowprops=dict(

            arrowstyle="->",

            lw=2,

            color="#555555"

        )

    )

    # ========================================================
    # Professional Grid
    # ========================================================

    ax.grid(

        linestyle="--",

        alpha=0.30

    )

    ax.set_axisbelow(True)

    # ========================================================
    # Axis Limits
    # ========================================================

    ax.set_xlim(-1,24)

    ax.set_ylim(98.0,100.0)

    ax.set_xticks(np.arange(0,25,5))

    ax.set_yticks(np.arange(98.0,100.1,0.25))

    # ========================================================
    # Clean Style
    # ========================================================

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(1.2)

    ax.spines["bottom"].set_linewidth(1.2)

    # ========================================================
    # Final Information Box
    # ========================================================

    info = (

        "Bubble Size : Model Size Reduction\n"

        "Bubble Color : Energy Reduction\n\n"

        "Final Model\n"

        "Accuracy : 99.43%\n"

        "MAC Reduction : 20.90%\n"

        "Energy Reduction : 28.43%\n"

        "Model Size Reduction : 73.84%"

    )

    ax.text(

        0.02,

        0.02,

        info,

        transform=ax.transAxes,

        ha="left",

        va="bottom",

        fontsize=9,

        bbox=dict(

            facecolor="white",

            edgecolor="gray",

            boxstyle="round,pad=0.45"

        )

    )

    # ========================================================
    # Figure Caption
    # ========================================================

    fig.text(

        0.5,

        0.02,

        "Bubble size represents model size reduction, while color intensity indicates "
        "energy reduction. The incrementally pruned model achieves the best overall "
        "balance between computational efficiency and classification accuracy.",

        ha="center",

        fontsize=10,

        style="italic"

    )

    # ========================================================
    # Layout
    # ========================================================

    plt.tight_layout(

        rect=[0,0.05,1,0.97]

    )

    # ========================================================
    # Save Figure
    # ========================================================

    save_figure(

        fig,

        "Figure_08_Multiobjective_Bubble_Plot"

    )

    plt.close()

    print("Figure 08 Completed.")

# ============================================================
# Figure 09
# Performance Improvement Dumbbell Plot (Clean Fixed Version)
# ============================================================

def generate_figure_09(df):

    print("\nGenerating Figure 09...")

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    metrics = [
        "Accuracy",
        "Model Size",
        "Parameters",
        "MACs",
        "Latency",
        "Energy",
        "CO₂"
    ]

    baseline_values = np.array([
        99.7238,        # %
        26.3835,        # MB
        2272550,        # params
        319004966,      # MACs
        34.7619,        # ms/image
        0.0012937816,   # kWh
        0.0008945348    # kgCO2eq
    ], dtype=float)

    final_values = np.array([
        99.4292,        # %
        6.9022,         # MB
        1737890,        # params
        249271775,      # MACs
        25.8211,        # ms/image
        0.0009330801,   # kWh
        0.0006451418    # kgCO2eq
    ], dtype=float)

    final_norm = (final_values / baseline_values) * 100.0
    y = np.arange(len(metrics))

    # --------------------------------------------------------
    # Helper formatting
    # --------------------------------------------------------

    def fmt_baseline(i, v):
        if i == 0:
            return f"{v:.4f}%"
        elif i == 1:
            return f"{v:.4f} MB"
        elif i in (2, 3):
            return f"{v/1e6:.2f}M"
        elif i == 4:
            return f"{v:.4f} ms"
        elif i == 5:
            return f"{v:.6f} kWh"
        else:
            return f"{v:.6f} kg"

    def fmt_final(i, v, norm):
        change = 100.0 - norm
        if i == 0:
            return f"{v:.4f}%\n({v - baseline_values[i]:+.4f} pp)"
        elif i == 1:
            return f"{v:.4f} MB\n({change:.2f}% smaller)"
        elif i in (2, 3):
            return f"{v/1e6:.2f}M\n({change:.2f}% fewer)"
        elif i == 4:
            return f"{v:.4f} ms\n({change:.2f}% faster)"
        elif i == 5:
            return f"{v:.6f} kWh\n({change:.2f}% lower)"
        else:
            return f"{v:.6f} kg\n({change:.2f}% lower)"

    # --------------------------------------------------------
    # Figure layout
    # --------------------------------------------------------

    fig = plt.figure(figsize=(17, 9.5), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[5.3, 1.7], wspace=0.06)

    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")

    # --------------------------------------------------------
    # Colors
    # --------------------------------------------------------

    baseline_color = "#9AA0A6"
    final_color = "#34A853"
    ring_color = "#D62728"
    line_color = "#7F8C8D"
    metric_color = "#222222"
    baseline_text_color = "#555555"
    final_text_color = "#1B5E20"

    # --------------------------------------------------------
    # Main dumbbell plot
    # --------------------------------------------------------

    ax.axvline(
        100,
        color="#444444",
        linestyle="--",
        linewidth=1.8,
        alpha=0.85,
        zorder=0
    )

    for i, yy in enumerate(y):
        x0 = 100.0
        x1 = float(final_norm[i])

        # connector
        ax.hlines(
            yy,
            xmin=min(x0, x1),
            xmax=max(x0, x1),
            color=line_color,
            linewidth=2.8,
            zorder=1
        )

        # baseline point
        ax.scatter(
            x0,
            yy,
            s=175,
            color=baseline_color,
            edgecolor="black",
            linewidth=1.2,
            zorder=3
        )

        # final point
        ax.scatter(
            x1,
            yy,
            s=245,
            color=final_color,
            edgecolor="black",
            linewidth=1.4,
            zorder=4
        )

        # highlight ring
        ax.scatter(
            x1,
            yy,
            s=480,
            facecolors="none",
            edgecolors=ring_color,
            linewidth=2.2,
            zorder=5
        )

        # metric labels on the far left
        ax.text(
            -43.0,
            yy,
            metrics[i],
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=metric_color,
            clip_on=False
        )

        # baseline values, kept left of the baseline line
        ax.text(
            96.0,
            yy,
            fmt_baseline(i, baseline_values[i]),
            ha="right",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color=baseline_text_color,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                boxstyle="round,pad=0.18",
                alpha=0.96
            ),
            clip_on=False
        )

        # final values, placed to the right
        ax.text(
            112.5,
            yy,
            fmt_final(i, final_values[i], x1),
            ha="left",
            va="center",
            fontsize=9.1,
            fontweight="bold",
            color=final_text_color,
            bbox=dict(
                facecolor="white",
                edgecolor="#C8E6C9",
                boxstyle="round,pad=0.28",
                alpha=0.98
            ),
            clip_on=False
        )

    ax.set_yticks(y)
    ax.set_yticklabels([""] * len(metrics))
    ax.invert_yaxis()

    ax.set_xlim(-46, 132)
    ax.set_xticks(np.arange(0, 121, 20))

    ax.set_xlabel(
        "Normalized Value (% of Baseline)",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_title(
        "Performance Improvement Dumbbell Plot",
        fontsize=18,
        fontweight="bold",
        pad=16
    )

    ax.grid(axis="x", linestyle="--", alpha=0.28)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ax.text(
        0.02,
        0.97,
        "Baseline is normalized to 100%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#444444"
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    from matplotlib.lines import Line2D

    legend_items = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=baseline_color,
            markeredgecolor="black",
            markersize=9,
            label="Baseline"
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=final_color,
            markeredgecolor="black",
            markersize=9,
            label="Final Optimized Model"
        )
    ]

    ax.legend(
        handles=legend_items,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.01),
        fontsize=10,
        frameon=True,
        fancybox=True,
        edgecolor="gray"
    )

    # --------------------------------------------------------
    # Right-side summary table
    # --------------------------------------------------------

    ax_info.text(
        0.50,
        0.97,
        "Baseline vs Final Summary",
        transform=ax_info.transAxes,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold"
    )

    table_rows = [
        ["Accuracy", "99.7238%", "99.4292%", "-0.2946 pp"],
        ["Model Size", "26.3835 MB", "6.9022 MB", "-73.84%"],
        ["Parameters", "2.27M", "1.74M", "-23.53%"],
        ["MACs", "319.00M", "249.27M", "-21.86%"],
        ["Latency", "34.7619 ms", "25.8211 ms", "-25.72%"],
        ["Energy", "0.0012938 kWh", "0.0009331 kWh", "-27.88%"],
        ["CO₂", "0.0008945 kg", "0.0006451 kg", "-27.88%"],
    ]

    col_labels = ["Metric", "Baseline", "Final", "Change"]

    table = ax_info.table(
        cellText=table_rows,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="upper center",
        bbox=[0.02, 0.22, 0.96, 0.68]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.7)
    table.scale(1.0, 1.26)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#B0B0B0")
        cell.set_linewidth(0.8)

        if r == 0:
            cell.set_facecolor("#EAF2FF")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_color("#1F1F1F")
        else:
            if c == 2:
                cell.set_facecolor("#E8F5E9")
            elif c == 3:
                cell.set_facecolor("#FFF3E0")
            else:
                cell.set_facecolor("white")

    note = (
        "Notes\n"
        "• Baseline is normalized to 100%.\n"
        "• Final model = safe pointwise\n"
        "  incremental-pruned FP32 model.\n"
        "• Exact values are shown in the table."
    )

    ax_info.text(
        0.03,
        0.15,
        note,
        transform=ax_info.transAxes,
        ha="left",
        va="top",
        fontsize=9.1,
        linespacing=1.35,
        bbox=dict(
            facecolor="white",
            edgecolor="gray",
            boxstyle="round,pad=0.40",
            alpha=0.98
        )
    )

    # --------------------------------------------------------
    # Caption
    # --------------------------------------------------------

    fig.text(
        0.5,
        0.015,
        "The dumbbell plot summarizes the before-after shift from the official baseline MobileNetV2 "
        "to the final safe pointwise incremental-pruned model.",
        ha="center",
        fontsize=10,
        style="italic"
    )

    # --------------------------------------------------------
    # Final layout + save
    # --------------------------------------------------------

    fig.subplots_adjust(
        top=0.90,
        bottom=0.08,
        left=0.06,
        right=0.98,
        wspace=0.05
    )

    save_figure(
        fig,
        "Figure_09_Performance_Improvement_Dumbbell"
    )

    plt.close(fig)

    print("Figure 09 Completed.")

# ============================================================
# Figure 10
# Multi-Metric Radar Chart Comparison
# Publication Quality Version
# ============================================================

def generate_figure_10(df):

    print("\nGenerating Figure 10...")

    # ========================================================
    # Data
    # ========================================================

    models = [
        "Baseline",
        "11% Structured",
        "Incremental"
    ]

    # Order selected for better visual interpretation
    metrics = [
        "Accuracy",
        "Energy\nReduction",
        "Latency\nReduction",
        "Model Size\nReduction",
        "MAC\nReduction",
        "Parameter\nReduction"
    ]

    # ========================================================
    # Experimental Results
    # ========================================================

    baseline = np.array([
        99.7238,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00
    ])

    structured = np.array([
        98.3800,
        27.78,
        26.21,
        72.72,
        20.16,
        20.57
    ])

    incremental = np.array([
        99.4292,
        27.88,
        25.72,
        73.84,
        21.86,
        23.53
    ])

    # ========================================================
    # Normalization
    # Each metric is normalized independently
    # ========================================================

    data = np.vstack([
        baseline,
        structured,
        incremental
    ])

    normalized = np.zeros_like(data)

    for j in range(data.shape[1]):

        column = data[:, j]

        max_value = np.max(column)

        if max_value == 0:

            normalized[:, j] = 0

        else:

            normalized[:, j] = (
                column / max_value
            ) * 100

    baseline_norm = normalized[0]
    structured_norm = normalized[1]
    incremental_norm = normalized[2]

    # ========================================================
    # Radar Coordinates
    # ========================================================

    N = len(metrics)

    angles = np.linspace(
        0,
        2 * np.pi,
        N,
        endpoint=False
    )

    angles = np.concatenate([
        angles,
        [angles[0]]
    ])

    baseline_plot = np.concatenate([
        baseline_norm,
        [baseline_norm[0]]
    ])

    structured_plot = np.concatenate([
        structured_norm,
        [structured_norm[0]]
    ])

    incremental_plot = np.concatenate([
        incremental_norm,
        [incremental_norm[0]]
    ])

    # ========================================================
    # Figure
    # ========================================================

    fig = plt.figure(
        figsize=(13.5, 11),
        dpi=300
    )

    ax = plt.subplot(
        111,
        polar=True
    )

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ========================================================
    # Colors
    # ========================================================

    baseline_color = "#8E8E93"

    structured_color = "#2F80ED"

    incremental_color = "#D62728"

    grid_color = "#D9D9D9"

    highlight_color = "#FFD700"

    # ========================================================
    # Polar Orientation
    # ========================================================

    ax.set_theta_offset(
        np.pi / 2
    )

    ax.set_theta_direction(
        -1
    )

    # ========================================================
    # Grid
    # ========================================================

    ax.set_ylim(0, 100)

    ax.set_yticks([
        20,
        40,
        60,
        80,
        100
    ])

    ax.set_yticklabels(
        [
            "20",
            "40",
            "60",
            "80",
            "100"
        ],
        fontsize=10,
        color="#666666"
    )

    ax.yaxis.grid(
        True,
        linestyle="--",
        linewidth=0.9,
        alpha=0.45,
        color=grid_color
    )

    ax.xaxis.grid(
        True,
        linestyle="-",
        linewidth=0.8,
        alpha=0.30,
        color=grid_color
    )

        # ========================================================
    # Publication-quality Radar Drawing Function
    # ========================================================

    def draw_model(values,
                   color,
                   label,
                   marker_size=8,
                   line_width=3,
                   alpha_fill=0.12,
                   z=5):

        # Polygon
        ax.plot(
            angles,
            values,
            color=color,
            linewidth=line_width,
            linestyle="-",
            solid_capstyle="round",
            label=label,
            zorder=z
        )

        # Fill
        ax.fill(
            angles,
            values,
            color=color,
            alpha=alpha_fill,
            zorder=z-1
        )

        # Markers
        ax.scatter(
            angles[:-1],
            values[:-1],
            s=marker_size**2,
            color="white",
            edgecolor=color,
            linewidth=2,
            zorder=z+2
        )


    # ========================================================
    # Draw Models
    # ========================================================

    draw_model(
        baseline_plot,
        baseline_color,
        "Baseline MobileNetV2",
        marker_size=7,
        line_width=2.8,
        alpha_fill=0.08,
        z=3
    )

    draw_model(
        structured_plot,
        structured_color,
        "11% Structured Pruning",
        marker_size=8,
        line_width=3,
        alpha_fill=0.10,
        z=4
    )

    draw_model(
        incremental_plot,
        incremental_color,
        "Incremental Pruning",
        marker_size=9,
        line_width=3.6,
        alpha_fill=0.18,
        z=6
    )

    # ========================================================
    # Highlight Incremental Model
    # ========================================================

    ax.scatter(
        angles[:-1],
        incremental_norm,
        s=170,
        color=highlight_color,
        edgecolor="black",
        linewidth=1.4,
        zorder=10
    )

    ax.scatter(
        angles[:-1],
        incremental_norm,
        s=52,
        color=incremental_color,
        edgecolor="none",
        zorder=11
    )

    # ========================================================
    # Best Metric Highlight Ring
    # ========================================================

    for ang, val in zip(angles[:-1], incremental_norm):

        ax.scatter(
            ang,
            val,
            s=290,
            facecolors="none",
            edgecolors=highlight_color,
            linewidth=1.8,
            zorder=9
        )

    # ========================================================
    # Metric Labels
    # ========================================================

    ax.set_xticks(angles[:-1])

    ax.set_xticklabels(
        metrics,
        fontsize=12,
        fontweight="bold",
        color="#202020"
    )

    # Move labels outward
    ax.tick_params(
        axis="x",
        pad=22
    )

    # ========================================================
    # Alternate Background Rings
    # ========================================================

    theta = np.linspace(0, 2*np.pi, 500)

    ring_levels = [20, 40, 60, 80, 100]

    ring_colors = [
        "#FAFAFA",
        "#F4F6F8",
        "#FAFAFA",
        "#F4F6F8",
        "#FAFAFA"
    ]

    previous = 0

    for level, color in zip(ring_levels, ring_colors):

        ax.fill_between(
            theta,
            previous,
            level,
            color=color,
            alpha=0.55,
            zorder=0
        )

        previous = level

    # ========================================================
    # Outer Border
    # ========================================================

    ax.spines["polar"].set_color("#888888")
    ax.spines["polar"].set_linewidth(1.4)

    # ========================================================
    # Title
    # ========================================================

    ax.set_title(
        "Multi-Metric Efficiency Profile of Optimized MobileNetV2 Models",
        fontsize=18,
        fontweight="bold",
        pad=34
    )

    # ========================================================
    # Professional Legend
    # ========================================================

    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.10, 1.02),
        fontsize=10.5,
        frameon=True,
        fancybox=True,
        framealpha=1,
        borderpad=0.8,
        labelspacing=0.8,
        handlelength=2.5
    )

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#808080")
    legend.get_frame().set_linewidth(1.2)

    # ========================================================
    # Best Model Annotation
    # ========================================================

    ax.text(
        0.50,
        -0.12,
        "★ Proposed Incremental Pruning achieves the best overall efficiency",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#B22222"
    )

    # ========================================================
    # Right-side Summary Box
    # ========================================================

    summary = (
        "Final Incremental Model\n"
        "────────────────────────\n"
        "Accuracy              : 99.4292%\n"
        "Parameter Reduction   : 23.53%\n"
        "MAC Reduction         : 21.86%\n"
        "Model Size Reduction  : 73.84%\n"
        "Latency Reduction     : 25.72%\n"
        "Energy Reduction      : 27.88%"
    )

    fig.text(
        0.82,
        0.24,
        summary,
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
        linespacing=1.55,
        bbox=dict(
            boxstyle="round,pad=0.55",
            facecolor="white",
            edgecolor="#D62728",
            linewidth=1.6
        )
    )

    # ========================================================
    # Caption
    # ========================================================

    fig.text(
        0.50,
        0.035,
        "Figure 10. Multi-metric radar comparison of the baseline MobileNetV2, "
        "11% structured pruning, and the proposed incremental pruning model. "
        "Values are independently normalized to facilitate comparison across "
        "metrics with different scales.",
        ha="center",
        fontsize=10,
        style="italic"
    )

    # ========================================================
    # Final Layout
    # ========================================================

    plt.tight_layout(
        rect=[0.02, 0.06, 0.80, 0.94]
    )

    # ========================================================
    # Save Figure
    # ========================================================

    save_figure(
        fig,
        "Figure_10_MultiMetric_Radar_Comparison"
    )

    plt.close(fig)

    print("Figure 10 Completed.")

# ============================================================
# Figure 11
# Correlation Matrix (Publication Quality)
# ============================================================

def generate_figure_11(df):

    print("\nGenerating Figure 11...")

    # ========================================================
    # Select only important metrics
    # ========================================================

    candidate_columns = [

        "pruning_ratio",
        "best_val_accuracy",
        "parameter_reduction_percent",
        "mac_reduction_percent",
        "accuracy_drop_percentage_point",
        "f1_drop_percentage_point",
        "fine_tune_time_seconds"

    ]

    # Keep only existing columns
    selected_columns = [
        c for c in candidate_columns
        if c in df.columns
    ]

    corr_df = df[selected_columns].copy()

    # ========================================================
    # Remove useless columns
    # ========================================================

    corr_df = corr_df.dropna(axis=1, how="all")

    corr_df = corr_df.loc[
        :,
        corr_df.nunique() > 1
    ]

    # ========================================================
    # Better display names
    # ========================================================

    rename_dict = {

        "pruning_ratio":
            "Pruning\nRatio",

        "best_val_accuracy":
            "Validation\nAccuracy",

        "parameter_reduction_percent":
            "Parameter\nReduction",

        "mac_reduction_percent":
            "MAC\nReduction",

        "accuracy_drop_percentage_point":
            "Accuracy\nDrop",

        "f1_drop_percentage_point":
            "F1\nDrop",

        "fine_tune_time_seconds":
            "Fine-tuning\nTime"

    }

    corr_df.rename(
        columns=rename_dict,
        inplace=True
    )

    # ========================================================
    # Correlation Matrix
    # ========================================================

    corr = corr_df.corr(numeric_only=True)

        # ========================================================
    # Figure Setup
    # ========================================================

    fig, ax = plt.subplots(

        figsize=(9, 8),

        dpi=300

    )

    # ========================================================
    # Heatmap
    # ========================================================

    import seaborn as sns

    sns.heatmap(

        corr,

        annot=True,

        fmt=".2f",

        cmap="RdBu_r",

        center=0,

        square=True,

        linewidths=0.8,

        linecolor="white",

        cbar=True,

        annot_kws={

            "fontsize":10,

            "fontweight":"bold"

        },

        cbar_kws={

            "shrink":0.85,

            "label":"Correlation Coefficient"

        },

        ax=ax

    )

    # ========================================================
    # Title
    # ========================================================

    ax.set_title(

        "Correlation Matrix of Key Evaluation Metrics",

        fontsize=18,

        fontweight="bold",

        pad=18

    )

    # ========================================================
    # Tick Labels
    # ========================================================

    ax.set_xticklabels(

        ax.get_xticklabels(),

        rotation=35,

        ha="right",

        fontsize=11,

        fontweight="bold"

    )

    ax.set_yticklabels(

        ax.get_yticklabels(),

        rotation=0,

        fontsize=11,

        fontweight="bold"

    )

        # ========================================================
    # Caption
    # ========================================================

    fig.text(

        0.5,

        0.02,

        "Figure 11. Pearson correlation matrix illustrating relationships among "
        "the key evaluation metrics of the proposed pruning framework.",

        ha="center",

        fontsize=10,

        style="italic"

    )

    # ========================================================
    # Layout
    # ========================================================

    plt.tight_layout(

        rect=[0.02, 0.05, 1.00, 0.96]

    )

    # ========================================================
    # Save Figure
    # ========================================================

    save_figure(

        fig,

        "Figure_11_Correlation_Matrix"

    )

    plt.close(fig)

    print("Figure 11 Completed.")

# ============================================================
# Figure 12
# Performance Evolution Across Progressive Pruning Stages
# ============================================================

def generate_figure_12(pruning_df):

    print("\nGenerating Figure 12...")

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    df = (
        pruning_df
        .sort_values("pruning_ratio")
        .reset_index(drop=True)
    )

    # X-axis labels
    stages = [
        "Baseline",
        "5%",
        "6%",
        "7%",
        "8%",
        "9%",
        "10%",
        "11%",
        "Incremental"
    ]

    x = np.arange(len(stages))

    # Baseline values
    baseline_accuracy = float(df["baseline_val_accuracy"].iloc[0])

    # --------------------------------------------------------
    # Actual experimental values
    # --------------------------------------------------------

    accuracy = np.array([
        baseline_accuracy,
        *df["best_val_accuracy"].tolist(),
        df["best_val_accuracy"].iloc[-1]
    ])

    parameter = np.array([
        0.0,
        *df["parameter_reduction_percent"].tolist(),
        df["parameter_reduction_percent"].iloc[-1]
    ])

    mac = np.array([
        0.0,
        *df["mac_reduction_percent"].tolist(),
        df["mac_reduction_percent"].iloc[-1]
    ])

    # --------------------------------------------------------
    # Create Figure
    # --------------------------------------------------------

    fig, ax1 = plt.subplots(figsize=(10, 5.8))

        # --------------------------------------------------------
    # Left Axis (Accuracy)
    # --------------------------------------------------------

    ax1.plot(

        x,
        accuracy,

        color=COLORS["blue"],

        marker="o",

        markersize=7,

        linewidth=2.8,

        label="Validation Accuracy"

    )

    ax1.set_ylabel(

        "Validation Accuracy (%)",

        fontsize=12,

        fontweight="bold",

        color=COLORS["blue"]

    )

    ax1.tick_params(

        axis="y",

        labelcolor=COLORS["blue"]

    )

    ax1.set_ylim(

        min(accuracy) - 0.5,
        max(accuracy) + 0.5

    )

    # --------------------------------------------------------
    # Right Axis (Reduction Metrics)
    # --------------------------------------------------------

    ax2 = ax1.twinx()

    ax2.plot(

        x,
        parameter,

        color=COLORS["green"],

        marker="s",

        markersize=7,

        linewidth=2.8,

        label="Parameter Reduction"

    )

    ax2.plot(

        x,
        mac,

        color=COLORS["red"],

        marker="^",

        markersize=7,

        linewidth=2.8,

        label="MAC Reduction"

    )

    ax2.set_ylabel(

        "Reduction (%)",

        fontsize=12,

        fontweight="bold",

        color=COLORS["red"]

    )

    ax2.tick_params(

        axis="y",

        labelcolor=COLORS["red"]

    )

    ax2.set_ylim(

        0,

        max(

            parameter.max(),
            mac.max()

        ) + 5

    )

    # --------------------------------------------------------
    # X-axis
    # --------------------------------------------------------

    ax1.set_xticks(x)

    ax1.set_xticklabels(

        stages,

        fontsize=11,

        fontweight="bold"

    )

    ax1.set_xlabel(

        "Optimization Stage",

        fontsize=12,

        fontweight="bold"

    )

    # --------------------------------------------------------
    # Grid & Title
    # --------------------------------------------------------

    ax1.grid(

        linestyle="--",

        alpha=0.30

    )

    ax1.set_axisbelow(True)

    ax1.set_title(

        "Performance Evolution Across Progressive Pruning Stages",

        fontsize=16,

        fontweight="bold",

        pad=12

    )

        # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(

        lines1 + lines2,

        labels1 + labels2,

        loc="upper left",

        fontsize=10,

        frameon=True,

        framealpha=0.95

    )

    # --------------------------------------------------------
    # Clean Axes
    # --------------------------------------------------------

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # --------------------------------------------------------
    # Tight Layout
    # --------------------------------------------------------

    plt.tight_layout()

    # --------------------------------------------------------
    # Save Figure
    # --------------------------------------------------------

    save_figure(

        fig,

        "Figure_12_Performance_Evolution"

    )

    plt.close(fig)

    print("Figure 12 Completed.")

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    pruning_df = load_pruning_csv()

    print("\nFramework Loaded Successfully.")

    generate_figure_01(pruning_df)
    generate_figure_02(pruning_df)
    generate_figure_03(pruning_df)
    generate_figure_04(pruning_df)
    generate_figure_05(pruning_df)
    generate_figure_06(pruning_df)
    generate_figure_07(pruning_df)
    generate_figure_08(pruning_df)
    generate_figure_09(pruning_df)
    generate_figure_10(pruning_df)
    generate_figure_11(pruning_df)
    generate_figure_12(pruning_df)
    

    print("\nAll completed.")