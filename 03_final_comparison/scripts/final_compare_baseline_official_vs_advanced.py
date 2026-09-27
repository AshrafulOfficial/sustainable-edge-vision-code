# ============================================================
# final_compare_baseline_official_vs_advanced.py
# Final Thesis Comparison Package
# Official Baseline MobileNetV2 vs Final Advanced Pruned FP32 MobileNetV2
#
# This fixed version:
# - uses official baseline summary for headline values
# - uses baseline /results folder for confusion matrix and class report
# - safely reads confusion matrix CSV with class-name column
# - generates fig_10/11 if baseline CM exists
# - generates fig_14/15 if both classification reports exist
# - generates fig_16/17 only if both raw latency CSV files exist
# - does not crash if optional files are missing
# ============================================================

import json
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. Path setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASELINE_OFFICIAL_DIR = (
    PROJECT_ROOT
    / "01_baseline_before_optimization"
    / "final_results"
    / "baseline_official_reference_final"
)

BASELINE_RESULTS_DIR = (
    PROJECT_ROOT
    / "01_baseline_before_optimization"
    / "results"
)

BASELINE_DETAILED_DIR = (
    PROJECT_ROOT
    / "01_baseline_before_optimization"
    / "final_results"
    / "baseline_final_cpu_bgd_offline_20260604_220258"
)

ADVANCED_FINAL_DIR = (
    PROJECT_ROOT
    / "02_after_pruning_quantization"
    / "advanced_pruning_experiments"
    / "final_results"
    / "advanced_pruned_final_cpu_bgd_offline_20260604_222049"
)

ADVANCED_DETAILED_DIR = ADVANCED_FINAL_DIR

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

RESULT_ROOT = (
    PROJECT_ROOT
    / "03_final_comparison"
    / "final_results"
    / f"baseline_official_vs_advanced_final_{RUN_ID}"
)

TABLE_DIR = RESULT_ROOT / "tables"
FIGURE_DIR = RESULT_ROOT / "paper_figures"
COPY_DIR = RESULT_ROOT / "copied_supporting_files"
NOTES_DIR = RESULT_ROOT / "notes"
LOG_DIR = RESULT_ROOT / "logs"

for d in [RESULT_ROOT, TABLE_DIR, FIGURE_DIR, COPY_DIR, NOTES_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("FINAL THESIS COMPARISON PACKAGE")
print("Official Baseline vs Final Advanced Pruned Model")
print("=" * 100)
print("Project root:", PROJECT_ROOT)
print("Result root:", RESULT_ROOT)


# ============================================================
# 2. Helper functions for file locating and reading
# ============================================================

def find_first_existing(candidates, required=False, label="file"):
    for p in candidates:
        if p.exists():
            return p
    if required:
        raise FileNotFoundError(
            f"Could not find required {label}. Checked:\n"
            + "\n".join(str(p) for p in candidates)
        )
    return None


def get_value(row, keys, required=True):
    for k in keys:
        if k in row and pd.notna(row[k]):
            return row[k]
    if required:
        raise KeyError(f"None of the keys found: {keys}")
    return None


def read_confusion_matrix_csv(path):
    """
    Read confusion matrix CSV safely.

    Some confusion matrix CSV files contain class names in first column.
    Example first cell: Apple___Apple_scab
    This function removes non-numeric label columns and keeps numeric matrix.
    """
    df = pd.read_csv(path)

    # Drop unnamed index columns if present.
    unnamed_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # If the first column is mostly non-numeric, it is likely class labels.
    if df.shape[1] > 0:
        first_col_numeric = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        if first_col_numeric.isna().mean() > 0.5:
            df = df.iloc[:, 1:]

    # Convert everything remaining to numeric.
    df = df.apply(pd.to_numeric, errors="coerce")

    # Remove all-empty rows/columns.
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if df.empty:
        raise ValueError(f"Confusion matrix became empty after parsing: {path}")

    return df.to_numpy(dtype=float)


def read_classification_report_csv(path):
    """
    Read sklearn classification_report CSV safely.

    Expected columns include:
    precision, recall, f1-score, support

    The class label may be stored as index column, unnamed first column,
    or a named column such as class_name.
    """
    df = pd.read_csv(path)

    # If first column looks like labels, use it as index.
    first_col = df.columns[0]
    expected_metric_cols = {"precision", "recall", "f1-score", "support"}

    if first_col not in expected_metric_cols:
        first_col_numeric = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        if first_col_numeric.isna().mean() > 0.5:
            df = df.set_index(first_col)

    # If still no f1-score, try index_col=0 fallback.
    if "f1-score" not in df.columns:
        df = pd.read_csv(path, index_col=0)

    if "f1-score" not in df.columns:
        raise ValueError(f"'f1-score' column not found in classification report: {path}")

    return df


def save_current_plot(filename_base):
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{filename_base}.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURE_DIR / f"{filename_base}.pdf", bbox_inches="tight")
    plt.close()


# ============================================================
# 3. Locate input files
# ============================================================

baseline_summary_csv = find_first_existing(
    [
        BASELINE_OFFICIAL_DIR / "baseline_final_all_metrics.csv",
        BASELINE_OFFICIAL_DIR / "latency_energy_co2" / "baseline_official_latency_energy_co2.csv",
        BASELINE_OFFICIAL_DIR / "metrics" / "baseline_official_metrics.csv",
    ],
    required=True,
    label="official baseline summary CSV",
)

advanced_summary_csv = find_first_existing(
    [
        ADVANCED_FINAL_DIR / "advanced_pruned_final_all_metrics.csv",
    ],
    required=True,
    label="advanced final summary CSV",
)

# Baseline detailed files: your actual files are in 01_baseline_before_optimization/results
baseline_report_csv = find_first_existing(
    [
        BASELINE_RESULTS_DIR / "baseline_classification_report.csv",
        BASELINE_RESULTS_DIR / "baseline_cpu_classification_report.csv",
        BASELINE_DETAILED_DIR / "metrics" / "baseline_classification_report.csv",
    ],
    required=False,
    label="baseline classification report",
)

baseline_cm_csv = find_first_existing(
    [
        BASELINE_RESULTS_DIR / "baseline_confusion_matrix.csv",
        BASELINE_RESULTS_DIR / "baseline_cpu_confusion_matrix.csv",
        BASELINE_DETAILED_DIR / "metrics" / "baseline_confusion_matrix.csv",
    ],
    required=False,
    label="baseline confusion matrix",
)

# Advanced detailed files
advanced_report_csv = find_first_existing(
    [
        ADVANCED_DETAILED_DIR / "metrics" / "advanced_pruned_classification_report.csv",
        PROJECT_ROOT
        / "02_after_pruning_quantization"
        / "advanced_pruning_experiments"
        / "results"
        / "safe_pointwise_kd_extra2_cpu_test_evaluation"
        / "safe_pointwise_kd_extra2_cpu_classification_report.csv",
    ],
    required=False,
    label="advanced classification report",
)

advanced_cm_csv = find_first_existing(
    [
        ADVANCED_DETAILED_DIR / "metrics" / "advanced_pruned_confusion_matrix.csv",
        PROJECT_ROOT
        / "02_after_pruning_quantization"
        / "advanced_pruning_experiments"
        / "results"
        / "safe_pointwise_kd_extra2_cpu_test_evaluation"
        / "safe_pointwise_kd_extra2_cpu_confusion_matrix.csv",
    ],
    required=False,
    label="advanced confusion matrix",
)

# Raw latency files
# Baseline raw latency was not found in your find output, but script checks possible locations.
baseline_latency_csv = find_first_existing(
    [
        BASELINE_RESULTS_DIR / "baseline_raw_forward_latency_values.csv",
        BASELINE_RESULTS_DIR / "baseline_raw_latency_values.csv",
        BASELINE_DETAILED_DIR / "latency_energy_co2" / "baseline_raw_forward_latency_values.csv",
        BASELINE_DETAILED_DIR / "latency_energy_co2" / "baseline_raw_latency_values.csv",
    ],
    required=False,
    label="baseline raw latency",
)

advanced_latency_csv = find_first_existing(
    [
        ADVANCED_DETAILED_DIR / "latency_energy_co2" / "advanced_pruned_raw_forward_latency_values.csv",
        PROJECT_ROOT
        / "02_after_pruning_quantization"
        / "advanced_pruning_experiments"
        / "results"
        / "safe_pointwise_kd_extra2_matched_baseline_process"
        / "safe_pointwise_kd_extra2_raw_latency_values.csv",
    ],
    required=False,
    label="advanced raw latency",
)

print("\nMain summary files found:")
print("Baseline official summary:", baseline_summary_csv)
print("Advanced final summary:", advanced_summary_csv)

print("\nOptional file check:")
optional_paths = {
    "Baseline classification report": baseline_report_csv,
    "Advanced classification report": advanced_report_csv,
    "Baseline confusion matrix": baseline_cm_csv,
    "Advanced confusion matrix": advanced_cm_csv,
    "Baseline raw latency": baseline_latency_csv,
    "Advanced raw latency": advanced_latency_csv,
}

for name, path in optional_paths.items():
    print(f"{name}: {'FOUND' if path is not None else 'MISSING'}")
    if path is not None:
        print("   ", path)


# ============================================================
# 4. Load main summary values
# ============================================================

baseline_df = pd.read_csv(baseline_summary_csv)
advanced_df = pd.read_csv(advanced_summary_csv)

baseline = baseline_df.iloc[0].to_dict()
advanced = advanced_df.iloc[0].to_dict()

baseline_values = {
    "model_label": "Official Baseline MobileNetV2",
    "short_label": "Baseline",
    "test_accuracy_percent": float(get_value(baseline, ["test_accuracy_percent"])),
    "macro_f1_percent": float(get_value(baseline, ["macro_f1_percent"])),
    "weighted_f1_percent": float(get_value(baseline, ["weighted_f1_percent"])),
    "model_size_mb": float(get_value(baseline, ["model_size_mb"])),
    "parameters": float(get_value(baseline, ["total_parameters", "parameters"])),
    "macs": float(get_value(baseline, ["macs"])),
    "avg_forward_latency_ms_per_image": float(get_value(baseline, ["avg_forward_latency_ms_per_image"])),
    "median_forward_latency_ms_per_image": float(get_value(baseline, ["median_forward_latency_ms_per_image"], required=False) or np.nan),
    "energy_consumed_kwh": float(get_value(baseline, ["energy_consumed_kwh"])),
    "energy_per_image_kwh": float(get_value(baseline, ["energy_per_image_kwh"])),
    "co2_emissions_kg": float(get_value(baseline, ["co2_emissions_kg"])),
    "co2_per_image_kg": float(get_value(baseline, ["co2_per_image_kg"])),
}

advanced_values = {
    "model_label": "Final Advanced Safe Pointwise Incremental-Pruned FP32 MobileNetV2",
    "short_label": "Advanced Pruned",
    "test_accuracy_percent": float(get_value(advanced, ["test_accuracy_percent"])),
    "macro_f1_percent": float(get_value(advanced, ["macro_f1_percent"])),
    "weighted_f1_percent": float(get_value(advanced, ["weighted_f1_percent"])),
    "model_size_mb": float(get_value(advanced, ["model_size_mb"])),
    "parameters": float(get_value(advanced, ["total_parameters", "parameters"])),
    "macs": float(get_value(advanced, ["macs"])),
    "avg_forward_latency_ms_per_image": float(get_value(advanced, ["avg_forward_latency_ms_per_image"])),
    "median_forward_latency_ms_per_image": float(get_value(advanced, ["median_forward_latency_ms_per_image"], required=False) or np.nan),
    "energy_consumed_kwh": float(get_value(advanced, ["energy_consumed_kwh"])),
    "energy_per_image_kwh": float(get_value(advanced, ["energy_per_image_kwh"])),
    "co2_emissions_kg": float(get_value(advanced, ["co2_emissions_kg"])),
    "co2_per_image_kg": float(get_value(advanced, ["co2_per_image_kg"])),
}

for row in [baseline_values, advanced_values]:
    row["parameters_million"] = row["parameters"] / 1_000_000
    row["macs_million"] = row["macs"] / 1_000_000
    row["energy_wh"] = row["energy_consumed_kwh"] * 1000
    row["co2_g"] = row["co2_emissions_kg"] * 1000
    row["energy_per_image_mwh"] = row["energy_per_image_kwh"] * 1_000_000
    row["co2_per_image_mg"] = row["co2_per_image_kg"] * 1_000_000


# ============================================================
# 5. Calculate reductions
# ============================================================

reduction_summary = {
    "accuracy_drop_pp": baseline_values["test_accuracy_percent"] - advanced_values["test_accuracy_percent"],
    "macro_f1_drop_pp": baseline_values["macro_f1_percent"] - advanced_values["macro_f1_percent"],
    "weighted_f1_drop_pp": baseline_values["weighted_f1_percent"] - advanced_values["weighted_f1_percent"],

    "model_size_reduction_percent": (
        (baseline_values["model_size_mb"] - advanced_values["model_size_mb"])
        / baseline_values["model_size_mb"]
    ) * 100,

    "parameter_reduction_percent": (
        (baseline_values["parameters"] - advanced_values["parameters"])
        / baseline_values["parameters"]
    ) * 100,

    "mac_reduction_percent": (
        (baseline_values["macs"] - advanced_values["macs"])
        / baseline_values["macs"]
    ) * 100,

    "latency_reduction_percent": (
        (baseline_values["avg_forward_latency_ms_per_image"] - advanced_values["avg_forward_latency_ms_per_image"])
        / baseline_values["avg_forward_latency_ms_per_image"]
    ) * 100,

    "energy_reduction_percent": (
        (baseline_values["energy_consumed_kwh"] - advanced_values["energy_consumed_kwh"])
        / baseline_values["energy_consumed_kwh"]
    ) * 100,

    "co2_reduction_percent": (
        (baseline_values["co2_emissions_kg"] - advanced_values["co2_emissions_kg"])
        / baseline_values["co2_emissions_kg"]
    ) * 100,
}


# ============================================================
# 6. Save final tables
# ============================================================

comparison_df = pd.DataFrame([baseline_values, advanced_values])
reduction_df = pd.DataFrame([reduction_summary])

comparison_df.to_csv(TABLE_DIR / "final_comparison_all_metrics.csv", index=False)
reduction_df.to_csv(TABLE_DIR / "final_reduction_summary.csv", index=False)

paper_ready_table = pd.DataFrame([
    {
        "Model": "Baseline MobileNetV2",
        "Test Accuracy (%)": baseline_values["test_accuracy_percent"],
        "Macro F1 (%)": baseline_values["macro_f1_percent"],
        "Weighted F1 (%)": baseline_values["weighted_f1_percent"],
        "Model Size (MB)": baseline_values["model_size_mb"],
        "Parameters": int(baseline_values["parameters"]),
        "MACs": int(baseline_values["macs"]),
        "CPU Latency (ms/image)": baseline_values["avg_forward_latency_ms_per_image"],
        "Energy (kWh)": baseline_values["energy_consumed_kwh"],
        "CO2 (kg)": baseline_values["co2_emissions_kg"],
    },
    {
        "Model": "Advanced Pruned FP32 MobileNetV2",
        "Test Accuracy (%)": advanced_values["test_accuracy_percent"],
        "Macro F1 (%)": advanced_values["macro_f1_percent"],
        "Weighted F1 (%)": advanced_values["weighted_f1_percent"],
        "Model Size (MB)": advanced_values["model_size_mb"],
        "Parameters": int(advanced_values["parameters"]),
        "MACs": int(advanced_values["macs"]),
        "CPU Latency (ms/image)": advanced_values["avg_forward_latency_ms_per_image"],
        "Energy (kWh)": advanced_values["energy_consumed_kwh"],
        "CO2 (kg)": advanced_values["co2_emissions_kg"],
    },
    {
        "Model": "Reduction / Change",
        "Test Accuracy (%)": -reduction_summary["accuracy_drop_pp"],
        "Macro F1 (%)": -reduction_summary["macro_f1_drop_pp"],
        "Weighted F1 (%)": -reduction_summary["weighted_f1_drop_pp"],
        "Model Size (MB)": reduction_summary["model_size_reduction_percent"],
        "Parameters": reduction_summary["parameter_reduction_percent"],
        "MACs": reduction_summary["mac_reduction_percent"],
        "CPU Latency (ms/image)": reduction_summary["latency_reduction_percent"],
        "Energy (kWh)": reduction_summary["energy_reduction_percent"],
        "CO2 (kg)": reduction_summary["co2_reduction_percent"],
    },
])

paper_ready_table.to_csv(TABLE_DIR / "final_paper_ready_table.csv", index=False)


# ============================================================
# 7. Copy supporting files if available
# ============================================================

copy_map = {
    baseline_summary_csv: COPY_DIR / "baseline_official_summary.csv",
    advanced_summary_csv: COPY_DIR / "advanced_final_summary.csv",
}

optional_support_files = [
    (baseline_report_csv, "baseline_classification_report.csv"),
    (advanced_report_csv, "advanced_classification_report.csv"),
    (baseline_cm_csv, "baseline_confusion_matrix.csv"),
    (advanced_cm_csv, "advanced_confusion_matrix.csv"),
    (baseline_latency_csv, "baseline_raw_latency.csv"),
    (advanced_latency_csv, "advanced_raw_latency.csv"),
]

for src, name in optional_support_files:
    if src is not None and src.exists():
        copy_map[src] = COPY_DIR / name

for src, dst in copy_map.items():
    shutil.copy2(src, dst)

print("\nCopied supporting files:", len(copy_map))


# ============================================================
# 8. Plot helper functions
# ============================================================

def grouped_bar(categories, baseline_vals, advanced_vals, ylabel, title, filename, fmt="{:.2f}"):
    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(9, 5))
    b1 = plt.bar(x - width / 2, baseline_vals, width, label="Baseline")
    b2 = plt.bar(x + width / 2, advanced_vals, width, label="Advanced Pruned")

    plt.xticks(x, categories, rotation=0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                fmt.format(h),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    save_current_plot(filename)


def horizontal_bar(labels, values, xlabel, title, filename, fmt="{:.2f}"):
    y = np.arange(len(labels))

    plt.figure(figsize=(8, 5))
    bars = plt.barh(y, values)
    plt.yticks(y, labels)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(axis="x", alpha=0.25)

    for bar, val in zip(bars, values):
        w = bar.get_width()
        plt.text(
            w,
            bar.get_y() + bar.get_height() / 2,
            " " + fmt.format(val),
            va="center",
            fontsize=9,
        )

    save_current_plot(filename)


def scatter_tradeoff(x_values, y_values, labels, xlabel, ylabel, title, filename):
    plt.figure(figsize=(7, 5))
    plt.scatter(x_values, y_values, s=130)

    for x, y, label in zip(x_values, y_values, labels):
        plt.annotate(label, (x, y), xytext=(7, 7), textcoords="offset points", fontsize=9)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.30)
    save_current_plot(filename)


def confusion_matrix_plot(cm, title, filename, normalize=False):
    cm_to_plot = cm.astype(float)

    if normalize:
        row_sums = cm_to_plot.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_to_plot = cm_to_plot / row_sums

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_to_plot, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted class index")
    plt.ylabel("True class index")
    plt.colorbar()
    save_current_plot(filename)


def latency_histogram(baseline_ms, advanced_ms, filename):
    plt.figure(figsize=(8, 5))
    plt.hist(baseline_ms, bins=40, alpha=0.60, label="Baseline raw latency")
    plt.hist(advanced_ms, bins=40, alpha=0.60, label="Advanced raw latency")
    plt.xlabel("Forward latency (ms/image)")
    plt.ylabel("Frequency")
    plt.title("Forward Latency Distribution Comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_current_plot(filename)


def latency_boxplot(baseline_ms, advanced_ms, filename):
    plt.figure(figsize=(7, 5))
    plt.boxplot([baseline_ms, advanced_ms], labels=["Baseline", "Advanced Pruned"], showfliers=False)
    plt.ylabel("Forward latency (ms/image)")
    plt.title("Forward Latency Boxplot Comparison")
    plt.grid(axis="y", alpha=0.25)
    save_current_plot(filename)


def read_latency_column(path):
    df = pd.read_csv(path)

    possible_cols = [
        "forward_latency_ms",
        "forward_latency_ms_per_image",
        "latency_ms",
        "latency_ms_per_image",
    ]

    for col in possible_cols:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").dropna().values

    # Fallback: first numeric column
    for col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() > 0:
            return vals.dropna().values

    raise ValueError(f"No numeric latency column found in {path}")


# ============================================================
# 9. Main final comparison figures
# ============================================================

grouped_bar(
    categories=["Accuracy", "Macro F1", "Weighted F1"],
    baseline_vals=[
        baseline_values["test_accuracy_percent"],
        baseline_values["macro_f1_percent"],
        baseline_values["weighted_f1_percent"],
    ],
    advanced_vals=[
        advanced_values["test_accuracy_percent"],
        advanced_values["macro_f1_percent"],
        advanced_values["weighted_f1_percent"],
    ],
    ylabel="Percentage (%)",
    title="Classification Performance Comparison",
    filename="fig_01_classification_performance_comparison",
    fmt="{:.2f}",
)

grouped_bar(
    categories=["Model Size (MB)", "Params (M)", "MACs (M)"],
    baseline_vals=[
        baseline_values["model_size_mb"],
        baseline_values["parameters_million"],
        baseline_values["macs_million"],
    ],
    advanced_vals=[
        advanced_values["model_size_mb"],
        advanced_values["parameters_million"],
        advanced_values["macs_million"],
    ],
    ylabel="Value",
    title="Compression and Complexity Comparison",
    filename="fig_02_compression_complexity_comparison",
    fmt="{:.2f}",
)

grouped_bar(
    categories=["Latency (ms)", "Energy (Wh)", "CO2 (g)"],
    baseline_vals=[
        baseline_values["avg_forward_latency_ms_per_image"],
        baseline_values["energy_wh"],
        baseline_values["co2_g"],
    ],
    advanced_vals=[
        advanced_values["avg_forward_latency_ms_per_image"],
        advanced_values["energy_wh"],
        advanced_values["co2_g"],
    ],
    ylabel="Converted value",
    title="Runtime Efficiency and Emission Comparison",
    filename="fig_03_latency_energy_co2_comparison",
    fmt="{:.4f}",
)

horizontal_bar(
    labels=[
        "Model size",
        "Parameters",
        "MACs",
        "Latency",
        "Energy",
        "CO2",
    ],
    values=[
        reduction_summary["model_size_reduction_percent"],
        reduction_summary["parameter_reduction_percent"],
        reduction_summary["mac_reduction_percent"],
        reduction_summary["latency_reduction_percent"],
        reduction_summary["energy_reduction_percent"],
        reduction_summary["co2_reduction_percent"],
    ],
    xlabel="Reduction compared with baseline (%)",
    title="Final Optimization Reductions",
    filename="fig_04_final_reduction_summary",
    fmt="{:.2f}%",
)

scatter_tradeoff(
    x_values=[
        baseline_values["test_accuracy_percent"],
        advanced_values["test_accuracy_percent"],
    ],
    y_values=[
        baseline_values["energy_consumed_kwh"],
        advanced_values["energy_consumed_kwh"],
    ],
    labels=["Baseline", "Advanced Pruned"],
    xlabel="Test accuracy (%)",
    ylabel="Total inference energy (kWh)",
    title="Accuracy-Energy Trade-off",
    filename="fig_05_accuracy_energy_tradeoff",
)

scatter_tradeoff(
    x_values=[
        baseline_values["test_accuracy_percent"],
        advanced_values["test_accuracy_percent"],
    ],
    y_values=[
        baseline_values["avg_forward_latency_ms_per_image"],
        advanced_values["avg_forward_latency_ms_per_image"],
    ],
    labels=["Baseline", "Advanced Pruned"],
    xlabel="Test accuracy (%)",
    ylabel="CPU forward latency (ms/image)",
    title="Accuracy-Latency Trade-off",
    filename="fig_06_accuracy_latency_tradeoff",
)

scatter_tradeoff(
    x_values=[
        baseline_values["test_accuracy_percent"],
        advanced_values["test_accuracy_percent"],
    ],
    y_values=[
        baseline_values["model_size_mb"],
        advanced_values["model_size_mb"],
    ],
    labels=["Baseline", "Advanced Pruned"],
    xlabel="Test accuracy (%)",
    ylabel="Saved model size (MB)",
    title="Accuracy-Model Size Trade-off",
    filename="fig_07_accuracy_model_size_tradeoff",
)

scatter_tradeoff(
    x_values=[
        baseline_values["test_accuracy_percent"],
        advanced_values["test_accuracy_percent"],
    ],
    y_values=[
        baseline_values["co2_emissions_kg"],
        advanced_values["co2_emissions_kg"],
    ],
    labels=["Baseline", "Advanced Pruned"],
    xlabel="Test accuracy (%)",
    ylabel="Estimated CO2 emission (kg)",
    title="Accuracy-CO2 Trade-off",
    filename="fig_08_accuracy_co2_tradeoff",
)

grouped_bar(
    categories=["Energy/Image (mWh)", "CO2/Image (mg)"],
    baseline_vals=[
        baseline_values["energy_per_image_mwh"],
        baseline_values["co2_per_image_mg"],
    ],
    advanced_vals=[
        advanced_values["energy_per_image_mwh"],
        advanced_values["co2_per_image_mg"],
    ],
    ylabel="Converted value per image",
    title="Per-image Energy and CO2 Comparison",
    filename="fig_09_per_image_energy_co2_comparison",
    fmt="{:.4f}",
)


# ============================================================
# 10. Confusion matrix figures
# ============================================================

if baseline_cm_csv is not None and baseline_cm_csv.exists():
    try:
        baseline_cm = read_confusion_matrix_csv(baseline_cm_csv)
        confusion_matrix_plot(
            baseline_cm,
            "Baseline Confusion Matrix",
            "fig_10_baseline_confusion_matrix",
            normalize=False,
        )
        confusion_matrix_plot(
            baseline_cm,
            "Baseline Normalized Confusion Matrix",
            "fig_11_baseline_confusion_matrix_normalized",
            normalize=True,
        )
        print("Generated baseline confusion matrix figures: fig_10, fig_11")
    except Exception as e:
        print("WARNING: Could not generate baseline confusion matrix figures:", e)
else:
    print("Skipping fig_10/fig_11: baseline confusion matrix CSV not found.")

if advanced_cm_csv is not None and advanced_cm_csv.exists():
    try:
        advanced_cm = read_confusion_matrix_csv(advanced_cm_csv)
        confusion_matrix_plot(
            advanced_cm,
            "Advanced Pruned Model Confusion Matrix",
            "fig_12_advanced_confusion_matrix",
            normalize=False,
        )
        confusion_matrix_plot(
            advanced_cm,
            "Advanced Pruned Model Normalized Confusion Matrix",
            "fig_13_advanced_confusion_matrix_normalized",
            normalize=True,
        )
        print("Generated advanced confusion matrix figures: fig_12, fig_13")
    except Exception as e:
        print("WARNING: Could not generate advanced confusion matrix figures:", e)
else:
    print("Skipping fig_12/fig_13: advanced confusion matrix CSV not found.")


# ============================================================
# 11. Per-class F1 comparison
# ============================================================

if (
    baseline_report_csv is not None
    and baseline_report_csv.exists()
    and advanced_report_csv is not None
    and advanced_report_csv.exists()
):
    try:
        baseline_report = read_classification_report_csv(baseline_report_csv)
        advanced_report = read_classification_report_csv(advanced_report_csv)

        ignore_rows = {"accuracy", "macro avg", "weighted avg", "micro avg"}

        common_rows = [
            str(idx) for idx in baseline_report.index
            if str(idx) in [str(x) for x in advanced_report.index]
            and str(idx) not in ignore_rows
        ]

        per_class_rows = []

        for cls in common_rows:
            b_f1 = float(baseline_report.loc[cls, "f1-score"])
            a_f1 = float(advanced_report.loc[cls, "f1-score"])

            per_class_rows.append({
                "class_name": cls,
                "baseline_f1": b_f1,
                "advanced_f1": a_f1,
                "baseline_f1_percent": b_f1 * 100,
                "advanced_f1_percent": a_f1 * 100,
                "f1_change_pp": (a_f1 - b_f1) * 100,
                "abs_f1_change_pp": abs((a_f1 - b_f1) * 100),
            })

        per_class_df = pd.DataFrame(per_class_rows)
        per_class_df.to_csv(TABLE_DIR / "per_class_f1_comparison.csv", index=False)

        if not per_class_df.empty:
            # Top 15 most changed classes
            top_change = per_class_df.sort_values("abs_f1_change_pp", ascending=False).head(15)
            top_change = top_change.sort_values("f1_change_pp")

            y = np.arange(len(top_change))

            plt.figure(figsize=(12, 7))
            plt.barh(y - 0.2, top_change["baseline_f1_percent"], height=0.4, label="Baseline")
            plt.barh(y + 0.2, top_change["advanced_f1_percent"], height=0.4, label="Advanced Pruned")
            plt.yticks(y, top_change["class_name"], fontsize=7)
            plt.xlabel("F1-score (%)")
            plt.title("Per-class F1 Comparison: 15 Most Changed Classes")
            plt.legend()
            plt.grid(axis="x", alpha=0.25)
            save_current_plot("fig_14_per_class_f1_top_changed")

            # Lowest 15 advanced class F1
            lowest_adv = per_class_df.sort_values("advanced_f1_percent", ascending=True).head(15)
            y = np.arange(len(lowest_adv))

            plt.figure(figsize=(12, 7))
            bars = plt.barh(y, lowest_adv["advanced_f1_percent"])
            plt.yticks(y, lowest_adv["class_name"], fontsize=7)
            plt.xlabel("Advanced model F1-score (%)")
            plt.title("Advanced Model Lowest 15 Per-class F1-scores")
            plt.grid(axis="x", alpha=0.25)

            for bar, val in zip(bars, lowest_adv["advanced_f1_percent"]):
                plt.text(val, bar.get_y() + bar.get_height() / 2, f" {val:.2f}", va="center", fontsize=8)

            save_current_plot("fig_15_advanced_lowest_15_class_f1")

            print("Generated per-class F1 figures: fig_14, fig_15")
        else:
            print("WARNING: Per-class F1 dataframe is empty; fig_14/fig_15 not generated.")

    except Exception as e:
        print("WARNING: Could not generate per-class F1 figures:", e)
else:
    print("Skipping fig_14/fig_15: both baseline and advanced classification reports are required.")


# ============================================================
# 12. Latency summary figures
# ============================================================

latency_note = ""

try:
    # Use official summary latency values for baseline and advanced.
    # This is safer than generating fake raw distributions.
    baseline_latency_stats = {
        "Average": baseline_values["avg_forward_latency_ms_per_image"],
        "Median": baseline_values["median_forward_latency_ms_per_image"],
    }

    advanced_latency_stats = {
        "Average": advanced_values["avg_forward_latency_ms_per_image"],
        "Median": advanced_values["median_forward_latency_ms_per_image"],
    }

    # If median is missing in any summary, use average as fallback.
    if np.isnan(baseline_latency_stats["Median"]):
        baseline_latency_stats["Median"] = baseline_latency_stats["Average"]

    if np.isnan(advanced_latency_stats["Median"]):
        advanced_latency_stats["Median"] = advanced_latency_stats["Average"]

    grouped_bar(
        categories=["Average latency", "Median latency"],
        baseline_vals=[
            baseline_latency_stats["Average"],
            baseline_latency_stats["Median"],
        ],
        advanced_vals=[
            advanced_latency_stats["Average"],
            advanced_latency_stats["Median"],
        ],
        ylabel="Latency (ms/image)",
        title="CPU Forward Latency Summary Comparison",
        filename="fig_16_latency_summary_comparison",
        fmt="{:.2f}",
    )

    # Separate simple bar for official average latency reduction
    plt.figure(figsize=(7, 5))
    labels = ["Baseline", "Advanced Pruned"]
    values = [
        baseline_values["avg_forward_latency_ms_per_image"],
        advanced_values["avg_forward_latency_ms_per_image"],
    ]

    bars = plt.bar(labels, values)
    plt.ylabel("Average CPU forward latency (ms/image)")
    plt.title("Official Average CPU Forward Latency Comparison")
    plt.grid(axis="y", alpha=0.25)

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    save_current_plot("fig_17_official_average_latency_comparison")

    latency_note = (
        "Latency figures were generated from official summary latency values. "
        "The baseline official CSV contains average, median, minimum, maximum, "
        "and standard deviation latency values, but not per-image raw latency values. "
        "Therefore, summary latency comparison figures were generated instead of "
        "raw latency histogram/boxplot figures."
    )

    print("Generated latency summary figures: fig_16, fig_17")

except Exception as e:
    latency_note = f"Could not generate latency summary figures: {e}"
    print("WARNING:", latency_note)

# ============================================================
# 13. Figure manifest
# ============================================================

figure_files = sorted([p.name for p in FIGURE_DIR.iterdir() if p.is_file()])

figure_manifest = pd.DataFrame({
    "figure_file": figure_files,
    "suggested_use": [
        "paper/thesis figure" if name.endswith(".png") else "editable/vector version"
        for name in figure_files
    ],
})

figure_manifest.to_csv(TABLE_DIR / "figure_manifest.csv", index=False)


# ============================================================
# 14. Notes and paper-ready writing
# ============================================================

methodology_note = f"""
Final Comparison Methodology
============================

This final comparison package compares the official before-optimization baseline MobileNetV2
with the final after-optimization advanced safe pointwise incremental-pruned FP32 MobileNetV2.

Main headline values:
- Baseline official summary source: {baseline_summary_csv}
- Advanced final summary source: {advanced_summary_csv}

Supporting files:
- Baseline classification report: {baseline_report_csv}
- Baseline confusion matrix: {baseline_cm_csv}
- Advanced classification report: {advanced_report_csv}
- Advanced confusion matrix: {advanced_cm_csv}
- Baseline raw latency: {baseline_latency_csv}
- Advanced raw latency: {advanced_latency_csv}

The headline comparison table should be used for final reported accuracy, model size,
parameters, MACs, latency, energy, and CO2 values.

Confusion matrices and per-class F1 plots were generated only when the required detailed
CSV files were available.

Latency note:
{latency_note}
"""

with open(NOTES_DIR / "methodology_note.txt", "w", encoding="utf-8") as f:
    f.write(methodology_note.strip() + "\n")

final_key_findings = f"""
Final Key Findings
==================

Official baseline MobileNetV2:
- Test accuracy: {baseline_values['test_accuracy_percent']:.4f}%
- Macro F1-score: {baseline_values['macro_f1_percent']:.4f}%
- Weighted F1-score: {baseline_values['weighted_f1_percent']:.4f}%
- Saved model size: {baseline_values['model_size_mb']:.4f} MB
- Parameters: {int(baseline_values['parameters']):,}
- MACs: {baseline_values['macs']:,.0f}
- Average CPU forward latency: {baseline_values['avg_forward_latency_ms_per_image']:.4f} ms/image
- Total inference energy: {baseline_values['energy_consumed_kwh']:.12f} kWh
- Estimated CO2 emission: {baseline_values['co2_emissions_kg']:.12f} kg

Final advanced pruned FP32 MobileNetV2:
- Test accuracy: {advanced_values['test_accuracy_percent']:.4f}%
- Macro F1-score: {advanced_values['macro_f1_percent']:.4f}%
- Weighted F1-score: {advanced_values['weighted_f1_percent']:.4f}%
- Saved model size: {advanced_values['model_size_mb']:.4f} MB
- Parameters: {int(advanced_values['parameters']):,}
- MACs: {advanced_values['macs']:,.0f}
- Average CPU forward latency: {advanced_values['avg_forward_latency_ms_per_image']:.4f} ms/image
- Total inference energy: {advanced_values['energy_consumed_kwh']:.12f} kWh
- Estimated CO2 emission: {advanced_values['co2_emissions_kg']:.12f} kg

Compared with the official baseline:
- Accuracy drop: {reduction_summary['accuracy_drop_pp']:.4f} percentage points
- Macro F1 drop: {reduction_summary['macro_f1_drop_pp']:.4f} percentage points
- Weighted F1 drop: {reduction_summary['weighted_f1_drop_pp']:.4f} percentage points
- Model-size reduction: {reduction_summary['model_size_reduction_percent']:.4f}%
- Parameter reduction: {reduction_summary['parameter_reduction_percent']:.4f}%
- MAC reduction: {reduction_summary['mac_reduction_percent']:.4f}%
- CPU latency reduction: {reduction_summary['latency_reduction_percent']:.4f}%
- Energy reduction: {reduction_summary['energy_reduction_percent']:.4f}%
- CO2 reduction: {reduction_summary['co2_reduction_percent']:.4f}%
"""

with open(NOTES_DIR / "final_key_findings.txt", "w", encoding="utf-8") as f:
    f.write(final_key_findings.strip() + "\n")

paper_paragraph = (
    f"Compared with the official baseline MobileNetV2, the final advanced safe pointwise "
    f"incremental-pruned FP32 MobileNetV2 achieved {advanced_values['test_accuracy_percent']:.2f}% "
    f"test accuracy and {advanced_values['macro_f1_percent']:.2f}% macro F1-score. "
    f"The optimized model introduced only a {reduction_summary['accuracy_drop_pp']:.2f} "
    f"percentage-point reduction in test accuracy and a {reduction_summary['macro_f1_drop_pp']:.2f} "
    f"percentage-point reduction in macro F1-score, while reducing saved model size by "
    f"{reduction_summary['model_size_reduction_percent']:.2f}%, parameters by "
    f"{reduction_summary['parameter_reduction_percent']:.2f}%, MACs by "
    f"{reduction_summary['mac_reduction_percent']:.2f}%, average CPU forward latency by "
    f"{reduction_summary['latency_reduction_percent']:.2f}%, total inference energy by "
    f"{reduction_summary['energy_reduction_percent']:.2f}%, and estimated CO2 emission by "
    f"{reduction_summary['co2_reduction_percent']:.2f}%."
)

with open(NOTES_DIR / "paper_ready_result_paragraph.txt", "w", encoding="utf-8") as f:
    f.write(paper_paragraph + "\n")

claim_points = """
Strict Thesis Claim Points
==========================

1. This thesis does not claim to introduce a new pruning algorithm.
2. The contribution is a practical Green AI and Edge AI evaluation of MobileNetV2-based crop disease detection.
3. MobileNetV2 was already lightweight, but the final advanced pruned FP32 model further reduced model size, parameters, MACs, CPU latency, energy, and CO2 emission.
4. The final model was selected based on the best practical trade-off between accuracy and efficiency, not accuracy alone.
5. The final advanced model should be described as selected 11% structured-pruned model plus additional 2% safe pointwise incremental pruning.
6. Do not report it as simple 13% pruning.
7. Report actual reductions relative to the official baseline: model size, parameters, MACs, latency, energy, and CO2.
"""

with open(NOTES_DIR / "thesis_claim_points.txt", "w", encoding="utf-8") as f:
    f.write(claim_points.strip() + "\n")


# ============================================================
# 15. Save config
# ============================================================

config = {
    "project_root": str(PROJECT_ROOT),
    "result_root": str(RESULT_ROOT),
    "baseline_official_dir": str(BASELINE_OFFICIAL_DIR),
    "baseline_summary_csv": str(baseline_summary_csv),
    "baseline_results_dir": str(BASELINE_RESULTS_DIR),
    "advanced_final_dir": str(ADVANCED_FINAL_DIR),
    "advanced_summary_csv": str(advanced_summary_csv),
    "baseline_report_csv": str(baseline_report_csv) if baseline_report_csv else None,
    "advanced_report_csv": str(advanced_report_csv) if advanced_report_csv else None,
    "baseline_cm_csv": str(baseline_cm_csv) if baseline_cm_csv else None,
    "advanced_cm_csv": str(advanced_cm_csv) if advanced_cm_csv else None,
    "baseline_latency_csv": str(baseline_latency_csv) if baseline_latency_csv else None,
    "advanced_latency_csv": str(advanced_latency_csv) if advanced_latency_csv else None,
    "latency_note": latency_note,
}

with open(RESULT_ROOT / "comparison_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)


# ============================================================
# 16. Print final summary
# ============================================================

print("\n" + "=" * 100)
print("FINAL COMPARISON PACKAGE CREATED SUCCESSFULLY")
print("=" * 100)
print("Result root:", RESULT_ROOT)
print("Tables:", TABLE_DIR)
print("Figures:", FIGURE_DIR)
print("Notes:", NOTES_DIR)
print("\nGenerated figures:")
for name in figure_files:
    print("-", name)
print("\nPaper-ready paragraph:")
print(paper_paragraph)
print("=" * 100)
