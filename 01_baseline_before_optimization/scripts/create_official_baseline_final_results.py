# ============================================================
# create_official_baseline_final_results.py
# Creates official baseline final result folder using verified official baseline values
# ============================================================

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASELINE_ROOT = PROJECT_ROOT / "01_baseline_before_optimization"
FINAL_RESULTS_ROOT = BASELINE_ROOT / "final_results"

OFFICIAL_DIR = FINAL_RESULTS_ROOT / "baseline_official_reference_final"
METRICS_DIR = OFFICIAL_DIR / "metrics"
FIGURES_DIR = OFFICIAL_DIR / "paper_figures"
LATENCY_DIR = OFFICIAL_DIR / "latency_energy_co2"

for d in [OFFICIAL_DIR, METRICS_DIR, FIGURES_DIR, LATENCY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Official verified baseline values
# ============================================================

official = {
    "model_key": "baseline_mobilenetv2",
    "model_name": "Official Baseline MobileNetV2",
    "model_status": "official_baseline_reference_final",
    "model_path": str(BASELINE_ROOT / "models" / "mobilenetv2_baseline_best.pth"),

    "test_images": 5431,
    "image_size": 224,
    "eval_batch_size": 32,
    "measure_batch_size": 1,
    "warmup_images": 50,
    "cpu_threads": 7,

    "model_size_mb": 26.383463859558105,
    "total_parameters": 2272550,
    "parameters_million": 2.272550,
    "macs": 319004966.0,
    "macs_million": 319.004966,

    "test_accuracy": 0.9972380777020806,
    "test_accuracy_percent": 99.72380777020805,

    "macro_precision": 0.9970410640122419,
    "macro_precision_percent": 99.70410640122419,
    "macro_recall": 0.9945161237895173,
    "macro_recall_percent": 99.45161237895173,
    "macro_f1": 0.9956856348556089,
    "macro_f1_percent": 99.56856348556089,

    "weighted_precision": 0.9972959405805608,
    "weighted_precision_percent": 99.72959405805608,
    "weighted_recall": 0.9972380777020806,
    "weighted_recall_percent": 99.72380777020805,
    "weighted_f1": 0.9972303794104317,
    "weighted_f1_percent": 99.72303794104317,

    "total_pipeline_time_seconds": 207.8583625830015,
    "avg_pipeline_time_seconds_per_image": 0.03827257642846649,

    "avg_forward_latency_seconds_per_image": 0.03476192660448368,
    "avg_forward_latency_ms_per_image": 34.761926604483676,
    "median_forward_latency_ms_per_image": 30.831670999759808,
    "min_forward_latency_ms_per_image": 21.856852001292282,
    "max_forward_latency_ms_per_image": 560.4564770001161,
    "std_forward_latency_ms_per_image": 17.269879291522116,

    "energy_consumed_kwh": 0.0012937816034293,
    "energy_per_image_kwh": 2.3822161727661572e-07,

    "co2_emissions_kg": 0.0008945348322086863,
    "co2_per_image_kg": 1.647090466228478e-07,

    "measurement_note": (
        "Official before-optimization baseline reference values. "
        "These values are used as the official baseline for final comparison. "
        "Classification metrics match the held-out test set evaluation. "
        "Latency, energy, and CO2 are from the official baseline latency-energy measurement."
    ),
}

# ============================================================
# Save CSV/JSON/TXT
# ============================================================

pd.DataFrame([official]).to_csv(OFFICIAL_DIR / "baseline_final_all_metrics.csv", index=False)
pd.DataFrame([official]).to_csv(METRICS_DIR / "baseline_official_metrics.csv", index=False)
pd.DataFrame([official]).to_csv(LATENCY_DIR / "baseline_official_latency_energy_co2.csv", index=False)

with open(OFFICIAL_DIR / "baseline_official_values.json", "w", encoding="utf-8") as f:
    json.dump(official, f, indent=2)

with open(OFFICIAL_DIR / "baseline_final_key_findings.txt", "w", encoding="utf-8") as f:
    f.write("Official Baseline MobileNetV2 Final Reference\n")
    f.write("=" * 80 + "\n")
    f.write(f"Test accuracy: {official['test_accuracy_percent']:.4f}%\n")
    f.write(f"Macro F1: {official['macro_f1_percent']:.4f}%\n")
    f.write(f"Weighted F1: {official['weighted_f1_percent']:.4f}%\n")
    f.write(f"Model size: {official['model_size_mb']:.4f} MB\n")
    f.write(f"Parameters: {official['total_parameters']:,}\n")
    f.write(f"MACs: {official['macs']:,.0f}\n")
    f.write(f"Average forward latency: {official['avg_forward_latency_ms_per_image']:.4f} ms/image\n")
    f.write(f"Median forward latency: {official['median_forward_latency_ms_per_image']:.4f} ms/image\n")
    f.write(f"Energy consumed: {official['energy_consumed_kwh']:.12f} kWh\n")
    f.write(f"Energy per image: {official['energy_per_image_kwh']:.12e} kWh/image\n")
    f.write(f"CO2 emissions: {official['co2_emissions_kg']:.12f} kg\n")
    f.write(f"CO2 per image: {official['co2_per_image_kg']:.12e} kg/image\n")
    f.write("\nReporting note:\n")
    f.write("This folder is treated as the official baseline reference for final comparison.\n")

# Also create a pointer file in final_results root
with open(FINAL_RESULTS_ROOT / "OFFICIAL_BASELINE_FINAL_PATH.txt", "w", encoding="utf-8") as f:
    f.write(str(OFFICIAL_DIR) + "\n")

# ============================================================
# Paper-ready single-model figures
# ============================================================

def save_bar(value, ylabel, title, filename, fmt="{:.2f}"):
    plt.figure(figsize=(6, 4))
    bars = plt.bar(["Baseline"], [value])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)

    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / f"{filename}.pdf", bbox_inches="tight")
    plt.close()

save_bar(official["test_accuracy_percent"], "Accuracy (%)", "Official Baseline Test Accuracy", "baseline_test_accuracy", "{:.2f}")
save_bar(official["macro_f1_percent"], "Macro F1-score (%)", "Official Baseline Macro F1-score", "baseline_macro_f1", "{:.2f}")
save_bar(official["model_size_mb"], "Saved model size (MB)", "Official Baseline Model Size", "baseline_model_size", "{:.2f}")
save_bar(official["parameters_million"], "Parameters (million)", "Official Baseline Parameters", "baseline_parameters", "{:.2f}")
save_bar(official["macs_million"], "MACs (million)", "Official Baseline MACs", "baseline_macs", "{:.2f}")
save_bar(official["avg_forward_latency_ms_per_image"], "Forward latency (ms/image)", "Official Baseline CPU Forward Latency", "baseline_forward_latency", "{:.2f}")
save_bar(official["energy_consumed_kwh"], "Total energy (kWh)", "Official Baseline Total Inference Energy", "baseline_total_energy", "{:.6f}")
save_bar(official["co2_emissions_kg"], "CO2 emission (kg)", "Official Baseline Estimated CO2 Emission", "baseline_co2_emission", "{:.8f}")
save_bar(official["energy_per_image_kwh"], "Energy per image (kWh/image)", "Official Baseline Energy per Image", "baseline_energy_per_image", "{:.2e}")
save_bar(official["co2_per_image_kg"], "CO2 per image (kg/image)", "Official Baseline CO2 per Image", "baseline_co2_per_image", "{:.2e}")

print("=" * 80)
print("Official baseline final results created successfully.")
print("Official folder:")
print(OFFICIAL_DIR)
print("Main file:")
print(OFFICIAL_DIR / "baseline_final_all_metrics.csv")
print("Figures:")
print(FIGURES_DIR)
print("=" * 80)