# ============================================================
# measure_safe_pointwise_kd_extra2_matched_baseline_process.py
# Advanced Safe Pointwise KD Pruned Candidate
# CPU Latency, Energy, and CO2 Measurement
# Matched to Baseline and 11% Pruned FP32 Process
# Thesis: Sustainable Edge Vision
# ============================================================

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from codecarbon import EmissionsTracker


# ============================================================
# 1. Path Setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "PlantVillage"

TEST_CSV = DATA_DIR / "test_files.csv"
LABEL_MAP_CSV = DATA_DIR / "label_map.csv"

ADV_ROOT = PROJECT_ROOT / "02_after_pruning_quantization" / "advanced_pruning_experiments"
MODEL_DIR = ADV_ROOT / "models"
RESULT_ROOT = ADV_ROOT / "results"

MODEL_PATH = MODEL_DIR / "safe_pointwise_kd_from_11p_extra_2percent_full_model_NOT_FINAL.pth"

RESULT_DIR = RESULT_ROOT / "safe_pointwise_kd_extra2_matched_baseline_process"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Configuration
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 1
NUM_WORKERS = 0
WARMUP_IMAGES = 50
SAMPLE_LIMIT = None

DEVICE = torch.device("cpu")

CPU_THREADS = max(1, os.cpu_count() - 1 if os.cpu_count() else 1)
torch.set_num_threads(CPU_THREADS)

print("=" * 90)
print("Advanced Safe Pointwise KD Extra-2% Candidate CPU Latency/Energy Measurement")
print("Matched to baseline and official 11% pruned FP32 measurement process")
print("=" * 90)
print("Project root:", PROJECT_ROOT)
print("Dataset path:", DATASET_PATH)
print("Test CSV:", TEST_CSV)
print("Model path:", MODEL_PATH)
print("Result folder:", RESULT_DIR)
print("Device:", DEVICE)
print("PyTorch version:", torch.__version__)
print("CPU threads:", torch.get_num_threads())
print("Image size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)
print("Warm-up images:", WARMUP_IMAGES)
print("Sample limit:", SAMPLE_LIMIT)


# ============================================================
# 3. File Checks
# ============================================================

required_paths = [
    DATASET_PATH,
    TEST_CSV,
    LABEL_MAP_CSV,
    MODEL_PATH,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(f"Missing required path: {path}")

print("\nAll required files found.")


# ============================================================
# 4. Load CSV Files
# ============================================================

test_df = pd.read_csv(TEST_CSV)

if SAMPLE_LIMIT is not None:
    test_df = test_df.head(SAMPLE_LIMIT).reset_index(drop=True)

label_map_df = pd.read_csv(LABEL_MAP_CSV).sort_values("label").reset_index(drop=True)

print("\nDataset information:")
print("Evaluation images:", len(test_df))
print("Number of classes:", len(label_map_df))


# ============================================================
# 5. Dataset Class
# ============================================================

class PlantVillageDataset(Dataset):
    def __init__(self, dataframe, dataset_root, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = self.dataset_root / row["relative_path"]
        label = int(row["label"])

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# 6. Transform and DataLoader
# ============================================================

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

test_dataset = PlantVillageDataset(
    dataframe=test_df,
    dataset_root=DATASET_PATH,
    transform=test_transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)

print("\nDataLoader information:")
print("Evaluation dataset size:", len(test_dataset))
print("Evaluation batches:", len(test_loader))


# ============================================================
# 7. Load Advanced Pruned Full Model
# ============================================================

print("\nLoading advanced pruned candidate model...")

try:
    loaded_obj = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )
except TypeError:
    loaded_obj = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

if isinstance(loaded_obj, torch.nn.Module):
    model = loaded_obj
elif isinstance(loaded_obj, dict) and "model" in loaded_obj:
    model = loaded_obj["model"]
else:
    raise RuntimeError(f"Unsupported model file format: {type(loaded_obj)}")

model = model.to(DEVICE)
model.eval()

model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
total_params = sum(p.numel() for p in model.parameters())

print("Model loaded successfully.")
print("Model size MB:", model_size_mb)
print("Total parameters:", total_params)


# ============================================================
# 8. Warm-up Inference
# ============================================================

print("\nRunning warm-up inference...")

warmup_count = 0

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        _ = model(images)

        warmup_count += images.size(0)

        if warmup_count >= WARMUP_IMAGES:
            break

print("Warm-up completed. Images used:", warmup_count)


# ============================================================
# 9. Measured CPU Inference with CodeCarbon
# ============================================================
# IMPORTANT:
# This measured loop intentionally matches the baseline and official
# 11% pruned FP32 matched-process latency-energy scripts.
#
# It does NOT calculate predictions, accuracy, F1-score, or classification report
# inside the CodeCarbon measured region.
#
# Accuracy/F1 should be measured separately using the CPU evaluation script.

print("\nStarting measured CPU inference...")

codecarbon_file = "safe_pointwise_kd_extra2_codecarbon_emissions.csv"

tracker = EmissionsTracker(
    output_dir=str(RESULT_DIR),
    output_file=codecarbon_file,
    project_name="Safe_Pointwise_KD_Extra2_Matched_CPU_Inference",
    log_level="error",
)

all_forward_latencies = []
total_images = 0

tracker.start()

pipeline_start_time = time.perf_counter()

with torch.inference_mode():
    for images, labels in tqdm(test_loader, desc="Measuring advanced candidate CPU inference"):
        images = images.to(DEVICE)

        forward_start_time = time.perf_counter()

        _ = model(images)

        forward_end_time = time.perf_counter()

        batch_forward_latency = forward_end_time - forward_start_time
        per_image_forward_latency = batch_forward_latency / images.size(0)

        all_forward_latencies.extend([per_image_forward_latency] * images.size(0))
        total_images += images.size(0)

pipeline_end_time = time.perf_counter()

emissions_kg = tracker.stop()

total_pipeline_time_seconds = pipeline_end_time - pipeline_start_time

latencies = np.array(all_forward_latencies)

avg_forward_latency_seconds = float(np.mean(latencies))
median_forward_latency_seconds = float(np.median(latencies))
min_forward_latency_seconds = float(np.min(latencies))
max_forward_latency_seconds = float(np.max(latencies))
std_forward_latency_seconds = float(np.std(latencies))


# ============================================================
# 10. Read CodeCarbon Energy Result
# ============================================================

codecarbon_csv = RESULT_DIR / codecarbon_file

energy_kwh = None

if codecarbon_csv.exists():
    emissions_df = pd.read_csv(codecarbon_csv)

    if "energy_consumed" in emissions_df.columns and len(emissions_df) > 0:
        energy_kwh = float(emissions_df["energy_consumed"].iloc[-1])

if energy_kwh is not None:
    energy_per_image_kwh = energy_kwh / total_images
else:
    energy_per_image_kwh = None

if emissions_kg is not None:
    co2_per_image_kg = emissions_kg / total_images
else:
    co2_per_image_kg = None


# ============================================================
# 11. Baseline and Current 11% Comparison Values
# ============================================================

baseline_model_size_mb = 26.383463859558105
baseline_latency_ms = 34.761926604483676
baseline_energy_kwh = 0.0012937816034293

current_11p_model_size_mb = 7.1973371505737305
current_11p_latency_ms = 25.651675099046948
current_11p_energy_kwh = 0.0009343486917365

latency_reduction_vs_baseline = ((baseline_latency_ms - avg_forward_latency_seconds * 1000) / baseline_latency_ms) * 100
model_size_reduction_vs_baseline = ((baseline_model_size_mb - model_size_mb) / baseline_model_size_mb) * 100

latency_change_vs_current_11p = ((current_11p_latency_ms - avg_forward_latency_seconds * 1000) / current_11p_latency_ms) * 100
model_size_reduction_vs_current_11p = ((current_11p_model_size_mb - model_size_mb) / current_11p_model_size_mb) * 100

if energy_kwh is not None:
    energy_reduction_vs_baseline = ((baseline_energy_kwh - energy_kwh) / baseline_energy_kwh) * 100
    energy_change_vs_current_11p = ((current_11p_energy_kwh - energy_kwh) / current_11p_energy_kwh) * 100
else:
    energy_reduction_vs_baseline = None
    energy_change_vs_current_11p = None


# ============================================================
# 12. Print Results
# ============================================================

print("\n" + "=" * 90)
print("Advanced Candidate CPU Latency, Energy, and CO2 Results")
print("=" * 90)
print("Total images:", total_images)

print("\nModel:")
print("Model size MB:", model_size_mb)
print("Total parameters:", total_params)

print("\nTime:")
print("Total pipeline time seconds:", total_pipeline_time_seconds)
print("Average pipeline time seconds/image:", total_pipeline_time_seconds / total_images)

print("\nForward latency:")
print("Average forward latency seconds/image:", avg_forward_latency_seconds)
print("Average forward latency ms/image:", avg_forward_latency_seconds * 1000)
print("Median forward latency ms/image:", median_forward_latency_seconds * 1000)
print("Minimum forward latency ms/image:", min_forward_latency_seconds * 1000)
print("Maximum forward latency ms/image:", max_forward_latency_seconds * 1000)
print("Std forward latency ms/image:", std_forward_latency_seconds * 1000)

print("\nEnergy and CO2:")
print("Energy consumed kWh:", energy_kwh)
print("Energy per image kWh:", energy_per_image_kwh)
print("CO2 emissions kg:", emissions_kg)
print("CO2 per image kg:", co2_per_image_kg)

print("\nComparison:")
print("Latency reduction vs baseline %:", latency_reduction_vs_baseline)
print("Energy reduction vs baseline %:", energy_reduction_vs_baseline)
print("Model-size reduction vs baseline %:", model_size_reduction_vs_baseline)
print("Latency change vs current 11% pruned %:", latency_change_vs_current_11p)
print("Energy change vs current 11% pruned %:", energy_change_vs_current_11p)
print("Model-size reduction vs current 11% pruned %:", model_size_reduction_vs_current_11p)


# ============================================================
# 13. Save Result CSV
# ============================================================

latency_energy_result = {
    "model_name": "safe_pointwise_kd_from_11p_extra_2percent",
    "model_status": "advanced_pruning_candidate_NOT_FINAL",
    "evaluation_device": "Dell Latitude CPU",
    "matched_to_scripts": "measure_baseline_latency_energy.py; measure_pruned_fp32_matched_baseline_process.py",
    "measurement_process_note": (
        "This advanced candidate measurement follows the same latency-energy measurement "
        "process used for the baseline and official 11% pruned FP32 matched-process scripts. "
        "It uses the same test CSV, preprocessing, image size, batch size, warm-up setting, "
        "CPU-thread rule, forward-latency timing boundary, and CodeCarbon tracker boundary. "
        "Accuracy/F1 calculation is not performed inside the measured loop."
    ),
    "model_path": str(MODEL_PATH),
    "total_images": total_images,
    "batch_size": BATCH_SIZE,
    "image_size": IMG_SIZE,
    "cpu_threads": torch.get_num_threads(),
    "model_size_mb": model_size_mb,
    "total_parameters": total_params,
    "total_pipeline_time_seconds": total_pipeline_time_seconds,
    "avg_pipeline_time_seconds_per_image": total_pipeline_time_seconds / total_images,
    "avg_forward_latency_seconds_per_image": avg_forward_latency_seconds,
    "avg_forward_latency_ms_per_image": avg_forward_latency_seconds * 1000,
    "median_forward_latency_ms_per_image": median_forward_latency_seconds * 1000,
    "min_forward_latency_ms_per_image": min_forward_latency_seconds * 1000,
    "max_forward_latency_ms_per_image": max_forward_latency_seconds * 1000,
    "std_forward_latency_ms_per_image": std_forward_latency_seconds * 1000,
    "energy_consumed_kwh": energy_kwh,
    "energy_per_image_kwh": energy_per_image_kwh,
    "co2_emissions_kg": emissions_kg,
    "co2_per_image_kg": co2_per_image_kg,
    "baseline_model_size_mb": baseline_model_size_mb,
    "baseline_avg_forward_latency_ms": baseline_latency_ms,
    "baseline_energy_kwh": baseline_energy_kwh,
    "current_11p_model_size_mb": current_11p_model_size_mb,
    "current_11p_avg_forward_latency_ms": current_11p_latency_ms,
    "current_11p_energy_kwh": current_11p_energy_kwh,
    "latency_reduction_percent_vs_baseline": latency_reduction_vs_baseline,
    "energy_reduction_percent_vs_baseline": energy_reduction_vs_baseline,
    "model_size_reduction_percent_vs_baseline": model_size_reduction_vs_baseline,
    "latency_change_percent_vs_current_11p": latency_change_vs_current_11p,
    "energy_change_percent_vs_current_11p": energy_change_vs_current_11p,
    "model_size_reduction_percent_vs_current_11p": model_size_reduction_vs_current_11p,
    "codecarbon_csv": str(codecarbon_csv),
    "note": (
        "Advanced pruning candidate matched-baseline latency-energy measurement. "
        "This result should be compared with both baseline and official 11% pruned FP32 results before final model selection."
    ),
}

result_df = pd.DataFrame([latency_energy_result])

result_path = RESULT_DIR / "safe_pointwise_kd_extra2_matched_latency_energy.csv"
result_df.to_csv(result_path, index=False)

raw_latency_path = RESULT_DIR / "safe_pointwise_kd_extra2_raw_latency_values.csv"
pd.DataFrame({
    "image_index": list(range(total_images)),
    "forward_latency_seconds": all_forward_latencies,
    "forward_latency_ms": np.array(all_forward_latencies) * 1000,
}).to_csv(raw_latency_path, index=False)

key_findings_path = RESULT_DIR / "safe_pointwise_kd_extra2_latency_energy_key_findings.txt"
with open(key_findings_path, "w", encoding="utf-8") as f:
    f.write("Advanced Safe Pointwise KD Extra-2% Candidate - Matched Latency/Energy Result\n")
    f.write("=" * 90 + "\n")
    f.write(f"Model size: {model_size_mb:.4f} MB\n")
    f.write(f"Parameters: {total_params:,}\n")
    f.write(f"Average forward latency: {avg_forward_latency_seconds * 1000:.4f} ms/image\n")
    f.write(f"Median forward latency: {median_forward_latency_seconds * 1000:.4f} ms/image\n")
    f.write(f"Energy consumed: {energy_kwh} kWh\n")
    f.write(f"Energy per image: {energy_per_image_kwh} kWh/image\n")
    f.write(f"CO2 emissions: {emissions_kg} kg\n")
    f.write(f"Latency reduction vs baseline: {latency_reduction_vs_baseline:.4f}%\n")
    f.write(f"Energy reduction vs baseline: {energy_reduction_vs_baseline}\n")
    f.write(f"Model-size reduction vs baseline: {model_size_reduction_vs_baseline:.4f}%\n")
    f.write(f"Latency change vs current 11% pruned: {latency_change_vs_current_11p:.4f}%\n")
    f.write(f"Energy change vs current 11% pruned: {energy_change_vs_current_11p}\n")
    f.write(f"Model-size reduction vs current 11% pruned: {model_size_reduction_vs_current_11p:.4f}%\n")

print("\nSaved files:")
print(result_path)
print(raw_latency_path)
print(codecarbon_csv)
print(key_findings_path)

print("\nAdvanced candidate CPU latency, energy, and CO2 measurement completed successfully.")