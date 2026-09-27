# ============================================================
# measure_pruned_fp32_matched_baseline_process.py
# 11% Structured-Pruned FP32 MobileNetV2
# CPU Latency, Energy, and CO2 Measurement
# Matched to: measure_baseline_latency_energy.py
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
from torchvision import transforms

from codecarbon import EmissionsTracker


# ============================================================
# 1. Path Setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "PlantVillage"

TEST_CSV = DATA_DIR / "test_files.csv"
LABEL_MAP_CSV = DATA_DIR / "label_map.csv"

OPT_DIR = PROJECT_ROOT / "02_after_pruning_quantization"
MODEL_DIR = OPT_DIR / "models"

# Separate result folder so old results are not overwritten
RESULT_DIR = OPT_DIR / "results"
MATCHED_RESULT_DIR = RESULT_DIR / "pruned_fp32_matched_baseline_process"

PRUNED_MODEL_PATH = MODEL_DIR / "mobilenetv2_pruned_best_full_model.pth"
BACKUP_PRUNED_MODEL_PATH = MODEL_DIR / "mobilenetv2_safe_structured_pruned_11_percent_full_model.pth"

if not PRUNED_MODEL_PATH.exists() and BACKUP_PRUNED_MODEL_PATH.exists():
    PRUNED_MODEL_PATH = BACKUP_PRUNED_MODEL_PATH

MATCHED_RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Configuration
# ============================================================

IMG_SIZE = 224

# Batch size 1 is used to match the baseline latency-energy script.
BATCH_SIZE = 1

NUM_WORKERS = 0

# Warm-up setting matched with the baseline latency-energy script.
WARMUP_IMAGES = 50

# For final thesis measurement, keep SAMPLE_LIMIT = None.
# For quick testing only, use SAMPLE_LIMIT = 100.
SAMPLE_LIMIT = None

DEVICE = torch.device("cpu")

CPU_THREADS = max(1, os.cpu_count() - 1 if os.cpu_count() else 1)
torch.set_num_threads(CPU_THREADS)

print("=" * 80)
print("11% Structured-Pruned FP32 MobileNetV2 CPU Latency, Energy, and CO2 Measurement")
print("Measurement process matched to baseline measure_baseline_latency_energy.py")
print("=" * 80)
print("Project root:", PROJECT_ROOT)
print("Dataset path:", DATASET_PATH)
print("Test CSV:", TEST_CSV)
print("Label map CSV:", LABEL_MAP_CSV)
print("Pruned model path:", PRUNED_MODEL_PATH)
print("Result folder:", MATCHED_RESULT_DIR)
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
    PRUNED_MODEL_PATH,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(f"Missing required path: {path}")

print("\nAll required files/folders found.")


# ============================================================
# 4. Load CSV Files
# ============================================================

test_df = pd.read_csv(TEST_CSV)
label_map_df = pd.read_csv(LABEL_MAP_CSV)

if SAMPLE_LIMIT is not None:
    test_df = test_df.head(SAMPLE_LIMIT).reset_index(drop=True)

label_map_df = label_map_df.sort_values("label").reset_index(drop=True)
num_classes = len(label_map_df)

print("\nDataset information:")
print("Evaluation images:", len(test_df))
print("Number of classes:", num_classes)


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
# 7. Load 11% Structured-Pruned FP32 Full Model
# ============================================================

print("\nLoading 11% structured-pruned FP32 model...")

try:
    loaded_obj = torch.load(
        PRUNED_MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )
except TypeError:
    loaded_obj = torch.load(
        PRUNED_MODEL_PATH,
        map_location=DEVICE,
    )

if isinstance(loaded_obj, torch.nn.Module):
    model = loaded_obj
elif isinstance(loaded_obj, dict) and "model" in loaded_obj:
    model = loaded_obj["model"]
elif isinstance(loaded_obj, dict) and "model_state_dict" in loaded_obj:
    raise RuntimeError(
        "This file contains only model_state_dict. For a structurally pruned model, "
        "use the full-model .pth file such as mobilenetv2_pruned_best_full_model.pth."
    )
else:
    raise RuntimeError(f"Unsupported model file format: {type(loaded_obj)}")

model = model.to(DEVICE)
model.eval()

model_size_mb = PRUNED_MODEL_PATH.stat().st_size / (1024 * 1024)
total_params = sum(p.numel() for p in model.parameters())

print("\nModel loaded successfully.")
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
# This measured loop intentionally matches the baseline latency-energy script.
# It does NOT calculate predictions, accuracy, F1-score, or classification report
# inside the CodeCarbon measured region. This keeps energy and pipeline timing
# more comparable with the baseline script.

print("\nStarting measured CPU inference...")

codecarbon_file = "pruned_fp32_matched_baseline_codecarbon_emissions.csv"

tracker = EmissionsTracker(
    output_dir=str(MATCHED_RESULT_DIR),
    output_file=codecarbon_file,
    project_name="Pruned_FP32_Matched_Baseline_CPU_Inference",
    log_level="error",
)

all_forward_latencies = []
total_images = 0

tracker.start()

pipeline_start_time = time.perf_counter()

with torch.inference_mode():
    for images, labels in tqdm(test_loader, desc="Measuring pruned FP32 CPU inference"):
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

codecarbon_csv = MATCHED_RESULT_DIR / codecarbon_file

energy_kwh = None

if codecarbon_csv.exists():
    emissions_df = pd.read_csv(codecarbon_csv)

    if "energy_consumed" in emissions_df.columns:
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
# 11. Print Results
# ============================================================

print("\n" + "=" * 80)
print("11% Pruned FP32 CPU Latency, Energy, and CO2 Results")
print("Measurement process matched to baseline")
print("=" * 80)
print("Total images:", total_images)

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


# ============================================================
# 12. Save Result CSV
# ============================================================

latency_energy_result = {
    "model_name": "11% Structured-Pruned FP32 MobileNetV2",
    "evaluation_device": "Dell Latitude 7390 CPU",
    "matched_to_script": "measure_baseline_latency_energy.py",
    "measurement_process_note": (
        "This pruned FP32 measurement script follows the baseline latency-energy "
        "measurement process. It uses the same test CSV, preprocessing, image size, "
        "batch size, warm-up setting, CPU threading rule, forward-latency timing "
        "boundary, and CodeCarbon tracker boundary. Accuracy/F1 calculation is not "
        "performed inside the measured loop."
    ),
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
    "codecarbon_csv": str(codecarbon_csv),
    "note": (
        "After-optimization 11% structured-pruned FP32 CPU inference measurement. "
        "Result is saved separately and does not overwrite previous no-RAPL or official results."
    ),
}

result_df = pd.DataFrame([latency_energy_result])

result_path = MATCHED_RESULT_DIR / "pruned_fp32_matched_baseline_latency_energy.csv"
result_df.to_csv(result_path, index=False)

print("\nPruned FP32 matched-baseline latency and energy result saved at:")
print(result_path)

print("\nCodeCarbon result saved at:")
print(codecarbon_csv)

print("\n11% pruned FP32 CPU latency, energy, and CO2 measurement completed successfully.")