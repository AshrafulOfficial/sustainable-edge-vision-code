# ============================================================
# sync_advanced_latency_energy_to_matched_baseline.py
#
# Purpose:
#   Keep "advanced_pruned_final_cpu_bgd_offline_20260604_222049" as the
#   single official-final folder (it holds accuracy/F1/confusion-matrix/
#   paper_figures that exist nowhere else), but overwrite ONLY its
#   latency / energy / CO2 fields with the values measured in
#   "safe_pointwise_kd_extra2_matched_baseline_process" (17:27 run),
#   which you have decided to treat as the correct measurement.
#
#   After syncing, the older matched_baseline_process folder is removed
#   since its data now lives inside the official final folder.
#
# Run this from anywhere; it uses PROJECT_ROOT below (edit if needed).
# ============================================================

import shutil
from pathlib import Path
import pandas as pd

# ---- EDIT THIS if your project root is different ----
PROJECT_ROOT = Path(__file__).resolve().parents[3]

OPT_DIR = PROJECT_ROOT / "02_after_pruning_quantization" / "advanced_pruning_experiments"

FINAL_DIR = OPT_DIR / "final_results" / "advanced_pruned_final_cpu_bgd_offline_20260604_222049"
MATCHED_DIR = OPT_DIR / "results" / "safe_pointwise_kd_extra2_matched_baseline_process"

# Source (the numbers you are keeping as official)
matched_csv = MATCHED_DIR / "safe_pointwise_kd_extra2_matched_latency_energy.csv"
matched_codecarbon = MATCHED_DIR / "safe_pointwise_kd_extra2_codecarbon_emissions.csv"
matched_raw_latency = MATCHED_DIR / "safe_pointwise_kd_extra2_raw_latency_values.csv"

# Targets inside the official final folder
target_all_metrics = FINAL_DIR / "advanced_pruned_final_all_metrics.csv"
target_latency_energy_co2 = FINAL_DIR / "latency_energy_co2" / "advanced_pruned_latency_energy_co2_bgd_offline.csv"
target_codecarbon = FINAL_DIR / "latency_energy_co2" / "advanced_pruned_codecarbon_bgd_offline.csv"
target_raw_latency = FINAL_DIR / "latency_energy_co2" / "advanced_pruned_raw_forward_latency_values.csv"
target_key_findings = FINAL_DIR / "advanced_pruned_final_key_findings.txt"

for p in [matched_csv, matched_codecarbon, matched_raw_latency,
          target_all_metrics, target_latency_energy_co2, target_key_findings]:
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}")

# ============================================================
# 1. Read the "correct" (matched_baseline_process) numbers
# ============================================================

src = pd.read_csv(matched_csv).iloc[0]

fields_to_copy = [
    "total_pipeline_time_seconds",
    "avg_pipeline_time_seconds_per_image",
    "avg_forward_latency_seconds_per_image",
    "avg_forward_latency_ms_per_image",
    "median_forward_latency_ms_per_image",
    "min_forward_latency_ms_per_image",
    "max_forward_latency_ms_per_image",
    "std_forward_latency_ms_per_image",
    "energy_consumed_kwh",
    "energy_per_image_kwh",
    "co2_emissions_kg",
    "co2_per_image_kg",
]

new_values = {f: src[f] for f in fields_to_copy}

print("Values that will be written into the official final folder:")
for k, v in new_values.items():
    print(f"  {k}: {v}")

# ============================================================
# 2. Patch advanced_pruned_final_all_metrics.csv
# ============================================================

df_all = pd.read_csv(target_all_metrics)
for f in fields_to_copy:
    if f in df_all.columns:
        df_all.at[0, f] = new_values[f]

# Recompute reduction percentages if baseline reference columns exist
BASELINE_LATENCY_MS = 34.761926604483676
BASELINE_ENERGY_KWH = 0.0012937816034293

if "avg_forward_latency_ms_per_image" in df_all.columns:
    latency_reduction = (BASELINE_LATENCY_MS - new_values["avg_forward_latency_ms_per_image"]) / BASELINE_LATENCY_MS * 100
    df_all["latency_reduction_percent_vs_baseline"] = latency_reduction if "latency_reduction_percent_vs_baseline" in df_all.columns else None
if "energy_consumed_kwh" in df_all.columns:
    energy_reduction = (BASELINE_ENERGY_KWH - new_values["energy_consumed_kwh"]) / BASELINE_ENERGY_KWH * 100
    if "energy_reduction_percent_vs_baseline" in df_all.columns:
        df_all["energy_reduction_percent_vs_baseline"] = energy_reduction

df_all.to_csv(target_all_metrics, index=False)
print(f"\nUpdated: {target_all_metrics}")

# ============================================================
# 3. Patch latency_energy_co2/advanced_pruned_latency_energy_co2_bgd_offline.csv
# ============================================================

df_lat = pd.read_csv(target_latency_energy_co2)
for f in fields_to_copy:
    if f in df_lat.columns:
        df_lat.at[0, f] = new_values[f]
df_lat.to_csv(target_latency_energy_co2, index=False)
print(f"Updated: {target_latency_energy_co2}")

# ============================================================
# 4. Replace codecarbon + raw latency CSVs wholesale (keep target filenames)
# ============================================================

shutil.copyfile(matched_codecarbon, target_codecarbon)
print(f"Replaced: {target_codecarbon}")

shutil.copyfile(matched_raw_latency, target_raw_latency)
print(f"Replaced: {target_raw_latency}")

# ============================================================
# 5. Patch the human-readable key findings text file
# ============================================================

key_findings_text = f"""Final Advanced Safe Pointwise Incremental-Pruned FP32 MobileNetV2 Measurement
==========================================================================================
Test accuracy: 99.4292%
Macro F1: 99.2262%
Weighted F1: 99.4295%
Model size: 6.9022 MB
Parameters: 1,737,890
MACs: 249,271,775
Average forward latency: {new_values['avg_forward_latency_ms_per_image']:.4f} ms/image
Energy consumed: {new_values['energy_consumed_kwh']:.9f} kWh
Energy per image: {new_values['energy_per_image_kwh']:.6e} kWh/image
CO2 emissions: {new_values['co2_emissions_kg']:.10f} kgCO2eq
CO2 per image: {new_values['co2_per_image_kg']:.6e} kgCO2eq/image

Measurement protocol:
- Dell Latitude CPU
- Held-out test set
- Batch size 1 for latency-energy-CO2
- 50 warm-up images
- CodeCarbon country_iso_code=BGD

Model description:
- Selected 11% structured-pruned model + additional 2% safe pointwise incremental pruning
- Do not report this as simple 13% pruning
- Report actual parameter, MAC, model-size, latency, energy, and CO2 reductions compared with baseline

Note: latency/energy/CO2 values synced from the 17:27 matched_baseline_process
measurement run, designated as the final official measurement.
"""

target_key_findings.write_text(key_findings_text)
print(f"Updated: {target_key_findings}")

# ============================================================
# 6. Remove the now-redundant older folder
# ============================================================

confirm = input(
    f"\nAbout to permanently delete:\n  {MATCHED_DIR}\nType 'yes' to confirm: "
)

if confirm.strip().lower() == "yes":
    shutil.rmtree(MATCHED_DIR)
    print(f"Deleted: {MATCHED_DIR}")
else:
    print("Skipped deletion. You can delete it manually later once you've verified everything.")

print("\nDone. The official final folder now uses the matched_baseline_process")
print("latency/energy/CO2 numbers, and the old duplicate folder is removed.")