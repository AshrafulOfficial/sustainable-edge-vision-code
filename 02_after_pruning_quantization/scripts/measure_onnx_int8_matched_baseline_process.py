# ============================================================
# measure_onnx_int8_matched_baseline_process.py
# ONNX INT8 Pruned MobileNetV2 11%
# CPU Latency, Energy, and CO2 Measurement
# Matched to: Baseline and 11% Pruned FP32 latency-energy process
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

import onnxruntime as ort
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
RESULT_DIR = OPT_DIR / "results"

ONNX_MODEL_PATH = MODEL_DIR / "mobilenetv2_pruned_11percent_onnx_int8.onnx"

MATCHED_RESULT_DIR = RESULT_DIR / "onnx_int8_matched_baseline_process_v2"
MATCHED_RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Configuration
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 1
NUM_WORKERS = 0
WARMUP_IMAGES = 50
SAMPLE_LIMIT = None
DEVICE_NAME = "cpu"

CPU_THREADS = max(1, os.cpu_count() - 1 if os.cpu_count() else 1)
torch.set_num_threads(CPU_THREADS)

print("=" * 80)
print("ONNX INT8 Pruned MobileNetV2 CPU Latency, Energy, and CO2 Measurement v2")
print("Measurement process matched to baseline and 11% pruned FP32 scripts")
print("=" * 80)
print("Project root         :", PROJECT_ROOT)
print("ONNX model path      :", ONNX_MODEL_PATH)
print("Result folder        :", MATCHED_RESULT_DIR)
print("PyTorch version      :", torch.__version__)
print("ONNX Runtime version :", ort.__version__)
print("CPU threads          :", CPU_THREADS)
print("Image size           :", IMG_SIZE)
print("Batch size           :", BATCH_SIZE)
print("Warm-up images       :", WARMUP_IMAGES)
print("Sample limit         :", SAMPLE_LIMIT)


# ============================================================
# 3. File Checks
# ============================================================

required_paths = [DATASET_PATH, TEST_CSV, LABEL_MAP_CSV, ONNX_MODEL_PATH]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(f"Missing required path: {path}")

print("\nAll required files/folders found.")


# ============================================================
# 4. Load CSV Files
# ============================================================

test_df      = pd.read_csv(TEST_CSV)
label_map_df = pd.read_csv(LABEL_MAP_CSV)

if SAMPLE_LIMIT is not None:
    test_df = test_df.head(SAMPLE_LIMIT).reset_index(drop=True)

label_map_df = label_map_df.sort_values("label").reset_index(drop=True)
num_classes  = len(label_map_df)

print("\nDataset information:")
print("Evaluation images :", len(test_df))
print("Number of classes :", num_classes)


# ============================================================
# 5. Dataset Class
# ============================================================

class PlantVillageDataset(Dataset):
    def __init__(self, dataframe, dataset_root, transform=None):
        self.dataframe    = dataframe.reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.transform    = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row        = self.dataframe.iloc[index]
        image_path = self.dataset_root / row["relative_path"]
        label      = int(row["label"])
        image      = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ============================================================
# 6. Transform and DataLoader
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

test_dataset = PlantVillageDataset(
    dataframe    = test_df,
    dataset_root = DATASET_PATH,
    transform    = test_transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    num_workers = NUM_WORKERS,
    pin_memory  = False,
)

print("\nDataLoader information:")
print("Evaluation dataset size :", len(test_dataset))
print("Evaluation batches      :", len(test_loader))


# ============================================================
# 7. Load ONNX INT8 Model
# ============================================================

print("\nLoading ONNX INT8 model...")

session_options = ort.SessionOptions()
session_options.intra_op_num_threads = CPU_THREADS
session_options.inter_op_num_threads = 1
session_options.execution_mode       = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession(
    str(ONNX_MODEL_PATH),
    sess_options = session_options,
    providers    = ["CPUExecutionProvider"],
)

input_name   = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]
providers    = session.get_providers()
model_size_mb = ONNX_MODEL_PATH.stat().st_size / (1024 * 1024)

print("Model loaded successfully.")
print("Input name   :", input_name)
print("Providers    :", providers)
print("Model size MB:", model_size_mb)


# ============================================================
# 8. Warm-up Inference
# ============================================================

print("\nRunning warm-up inference...")

warmup_count = 0

for images, labels in test_loader:
    # Same as baseline: no extra conversion
    images_np = images.numpy()

    _ = session.run(None, {input_name: images_np})

    warmup_count += images.size(0)
    if warmup_count >= WARMUP_IMAGES:
        break

print("Warm-up completed. Images used:", warmup_count)


# ============================================================
# 9. Measured CPU Inference with CodeCarbon
# ============================================================

print("\nStarting measured CPU inference...")

codecarbon_file = "onnx_int8_matched_v2_codecarbon_emissions.csv"

tracker = EmissionsTracker(
    output_dir   = str(MATCHED_RESULT_DIR),
    output_file  = codecarbon_file,
    project_name = "ONNX_INT8_Matched_Baseline_CPU_Inference_v2",
    log_level    = "error",
)

all_forward_latencies = []
total_images          = 0

tracker.start()
pipeline_start_time = time.perf_counter()

for images, labels in tqdm(test_loader, desc="Measuring ONNX INT8 CPU inference"):
    # Removed ascontiguousarray and astype — same memory path as baseline
    images_np = images.numpy()

    forward_start_time = time.perf_counter()
    _ = session.run(None, {input_name: images_np})
    forward_end_time = time.perf_counter()

    batch_forward_latency    = forward_end_time - forward_start_time
    per_image_forward_latency = batch_forward_latency / images.size(0)

    all_forward_latencies.extend([per_image_forward_latency] * images.size(0))
    total_images += images.size(0)

pipeline_end_time = time.perf_counter()
emissions_kg      = tracker.stop()

total_pipeline_time_seconds = pipeline_end_time - pipeline_start_time
latencies                   = np.array(all_forward_latencies)

avg_forward_latency_seconds    = float(np.mean(latencies))
median_forward_latency_seconds = float(np.median(latencies))
min_forward_latency_seconds    = float(np.min(latencies))
max_forward_latency_seconds    = float(np.max(latencies))
std_forward_latency_seconds    = float(np.std(latencies))


# ============================================================
# 10. Read CodeCarbon Energy Result
# ============================================================

codecarbon_csv = MATCHED_RESULT_DIR / codecarbon_file
energy_kwh     = None

if codecarbon_csv.exists():
    emissions_df = pd.read_csv(codecarbon_csv)
    if "energy_consumed" in emissions_df.columns and len(emissions_df) > 0:
        energy_kwh = float(emissions_df["energy_consumed"].iloc[-1])

energy_per_image_kwh = energy_kwh / total_images if energy_kwh is not None else None
co2_per_image_kg     = emissions_kg / total_images if emissions_kg is not None else None


# ============================================================
# 11. Print Results
# ============================================================

print("\n" + "=" * 80)
print("ONNX INT8 CPU Latency, Energy, and CO2 Results v2")
print("=" * 80)
print("Total images                      :", total_images)
print("Model size MB                     :", model_size_mb)
print("Total pipeline time seconds       :", total_pipeline_time_seconds)
print("Avg pipeline time ms/image        :", (total_pipeline_time_seconds / total_images) * 1000)
print("Avg forward latency ms/image      :", avg_forward_latency_seconds * 1000)
print("Median forward latency ms/image   :", median_forward_latency_seconds * 1000)
print("Min forward latency ms/image      :", min_forward_latency_seconds * 1000)
print("Max forward latency ms/image      :", max_forward_latency_seconds * 1000)
print("Std forward latency ms/image      :", std_forward_latency_seconds * 1000)
print("Energy consumed kWh               :", energy_kwh)
print("Energy per image kWh              :", energy_per_image_kwh)
print("CO2 emissions kg                  :", emissions_kg)
print("CO2 per image kg                  :", co2_per_image_kg)


# ============================================================
# 12. Save Result CSV
# ============================================================

latency_energy_result = {
    "model_name"                          : "ONNX INT8 Pruned MobileNetV2 11% v2",
    "evaluation_device"                   : "Dell Latitude 7390 CPU",
    "runtime"                             : "ONNX Runtime CPUExecutionProvider",
    "change_vs_v1"                        : "Removed ascontiguousarray and astype conversion to match baseline memory path",
    "total_images"                        : total_images,
    "batch_size"                          : BATCH_SIZE,
    "image_size"                          : IMG_SIZE,
    "cpu_threads"                         : CPU_THREADS,
    "model_size_mb"                       : model_size_mb,
    "total_pipeline_time_seconds"         : total_pipeline_time_seconds,
    "avg_pipeline_time_ms_per_image"      : (total_pipeline_time_seconds / total_images) * 1000,
    "avg_forward_latency_ms_per_image"    : avg_forward_latency_seconds * 1000,
    "median_forward_latency_ms_per_image" : median_forward_latency_seconds * 1000,
    "min_forward_latency_ms_per_image"    : min_forward_latency_seconds * 1000,
    "max_forward_latency_ms_per_image"    : max_forward_latency_seconds * 1000,
    "std_forward_latency_ms_per_image"    : std_forward_latency_seconds * 1000,
    "energy_consumed_kwh"                 : energy_kwh,
    "energy_per_image_kwh"                : energy_per_image_kwh,
    "co2_emissions_kg"                    : emissions_kg,
    "co2_per_image_kg"                    : co2_per_image_kg,
    "codecarbon_csv"                      : str(codecarbon_csv),
}

result_path = MATCHED_RESULT_DIR / "onnx_int8_matched_v2_latency_energy.csv"
pd.DataFrame([latency_energy_result]).to_csv(result_path, index=False)

raw_latency_path = MATCHED_RESULT_DIR / "onnx_int8_matched_v2_raw_latency_values.csv"
pd.DataFrame({
    "image_index"            : list(range(total_images)),
    "forward_latency_seconds": all_forward_latencies,
    "forward_latency_ms"     : np.array(all_forward_latencies) * 1000,
}).to_csv(raw_latency_path, index=False)

print("\nResults saved :", result_path)
print("Raw latency   :", raw_latency_path)
print("CodeCarbon    :", codecarbon_csv)
print("\nONNX INT8 CPU measurement v2 completed successfully.")