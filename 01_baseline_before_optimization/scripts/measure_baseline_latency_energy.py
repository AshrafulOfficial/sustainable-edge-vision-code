# ============================================================
# measure_baseline_latency_energy.py
# Baseline MobileNetV2 CPU Latency, Energy, and CO2 Measurement
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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from codecarbon import EmissionsTracker


# ============================================================
# 1. Path Setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "PlantVillage"

TEST_CSV = DATA_DIR / "test_files.csv"
LABEL_MAP_CSV = DATA_DIR / "label_map.csv"

BASELINE_DIR = PROJECT_ROOT / "01_baseline_before_optimization"
MODEL_DIR = BASELINE_DIR / "models"
RESULT_DIR = BASELINE_DIR / "results"

BEST_MODEL_PATH = MODEL_DIR / "mobilenetv2_baseline_best.pth"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Configuration
# ============================================================

IMG_SIZE = 224

# Batch size 1 is more realistic for edge-style single image inference.
# Use the same setting later for optimized model measurement.
BATCH_SIZE = 1

NUM_WORKERS = 0

# Warm-up helps stabilize latency measurement.
WARMUP_IMAGES = 50

# For final thesis measurement, keep SAMPLE_LIMIT = None.
# For quick testing only, use SAMPLE_LIMIT = 100.
SAMPLE_LIMIT = None

DEVICE = torch.device("cpu")

CPU_THREADS = max(1, os.cpu_count() - 1 if os.cpu_count() else 1)
torch.set_num_threads(CPU_THREADS)

print("=" * 80)
print("Baseline MobileNetV2 CPU Latency, Energy, and CO2 Measurement")
print("=" * 80)
print("Project root:", PROJECT_ROOT)
print("Dataset path:", DATASET_PATH)
print("Test CSV:", TEST_CSV)
print("Label map CSV:", LABEL_MAP_CSV)
print("Best model path:", BEST_MODEL_PATH)
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
    BEST_MODEL_PATH,
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
# 7. Load Baseline Model
# ============================================================

model = models.mobilenet_v2(weights=None)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

model_size_mb = BEST_MODEL_PATH.stat().st_size / (1024 * 1024)
total_params = sum(p.numel() for p in model.parameters())

print("\nModel loaded successfully.")
print("Checkpoint epoch:", checkpoint.get("epoch"))
print("Best validation accuracy:", checkpoint.get("best_val_accuracy"))
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

print("\nStarting measured CPU inference...")

tracker = EmissionsTracker(
    output_dir=str(RESULT_DIR),
    output_file="baseline_codecarbon_emissions.csv",
    project_name="MobileNetV2_Baseline_CPU_Inference",
    log_level="error",
)

all_forward_latencies = []
total_images = 0

tracker.start()

pipeline_start_time = time.perf_counter()

with torch.inference_mode():
    for images, labels in tqdm(test_loader, desc="Measuring baseline CPU inference"):
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

codecarbon_csv = RESULT_DIR / "baseline_codecarbon_emissions.csv"

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
print("Baseline CPU Latency, Energy, and CO2 Results")
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
    "model_name": "MobileNetV2 Baseline",
    "evaluation_device": "Dell Latitude 7390 CPU",
    "checkpoint_epoch": checkpoint.get("epoch"),
    "best_validation_accuracy": checkpoint.get("best_val_accuracy"),
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
    "note": "Baseline CPU inference measurement before pruning and quantization. Same measurement setup must be used for optimized model.",
}

result_df = pd.DataFrame([latency_energy_result])

result_path = RESULT_DIR / "baseline_latency_energy.csv"
result_df.to_csv(result_path, index=False)

print("\nBaseline latency and energy result saved at:")
print(result_path)

print("\nBaseline CPU latency, energy, and CO2 measurement completed successfully.")