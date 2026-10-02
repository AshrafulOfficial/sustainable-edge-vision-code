# ============================================================
# final_measure_advanced_pruned_cpu_bgd_offline.py
# Final Advanced Safe Pointwise Incremental-Pruned FP32 MobileNetV2 CPU Measurement
# Saves all final advanced-model values, CO2, and paper-ready figures
# Thesis: Sustainable Edge Vision
# ============================================================

import os
import time
import json
import platform
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt

from codecarbon import OfflineEmissionsTracker

try:
    import torch_pruning as tp
    TORCH_PRUNING_AVAILABLE = True
except Exception:
    tp = None
    TORCH_PRUNING_AVAILABLE = False


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path("/home/ashraful/Documents/Final year Thesis/thesis_project")

ADV_ROOT = PROJECT_ROOT / "02_after_pruning_quantization" / "advanced_pruning_experiments"
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "PlantVillage"
TEST_CSV = DATA_DIR / "test_files.csv"
LABEL_MAP_CSV = DATA_DIR / "label_map.csv"

MODEL_PATH = ADV_ROOT.parent / "models" / "mobilenetv2_safe_structured_pruned_11_percent_full_model.pth"

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_ROOT = ADV_ROOT / "final_results" / f"structured_11pct_cpu_bgd_offline_{RUN_ID}"

METRICS_DIR = RESULT_ROOT / "metrics"
FIGURES_DIR = RESULT_ROOT / "paper_figures"
LATENCY_DIR = RESULT_ROOT / "latency_energy_co2"
LOGS_DIR = RESULT_ROOT / "logs"

for folder in [METRICS_DIR, FIGURES_DIR, LATENCY_DIR, LOGS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Configuration
# ============================================================

DEVICE = torch.device("cpu")
IMG_SIZE = 224

EVAL_BATCH_SIZE = 32
MEASURE_BATCH_SIZE = 1

NUM_WORKERS = 0
WARMUP_IMAGES = 50
SAMPLE_LIMIT = None

CPU_THREADS = max(1, (os.cpu_count() or 2) - 1)
torch.set_num_threads(CPU_THREADS)

COUNTRY_ISO_CODE = "BGD"

MODEL_KEY = "advanced_safe_pointwise_incremental_pruned"
MODEL_NAME = "Advanced Safe Pointwise Incremental-Pruned FP32 MobileNetV2"

# Verified from Colab and Dell evaluation:
KNOWN_MACS = 254701367.0

print("=" * 100)
print("FINAL ADVANCED PRUNED MODEL CPU MEASUREMENT")
print("=" * 100)
print("Project root:", PROJECT_ROOT)
print("Model path:", MODEL_PATH)
print("Result root:", RESULT_ROOT)
print("Dataset path:", DATASET_PATH)
print("Test CSV:", TEST_CSV)
print("Label map CSV:", LABEL_MAP_CSV)
print("Device:", DEVICE)
print("PyTorch:", torch.__version__)
print("Python:", platform.python_version())
print("CPU threads:", torch.get_num_threads())
print("Evaluation batch size:", EVAL_BATCH_SIZE)
print("Measurement batch size:", MEASURE_BATCH_SIZE)
print("Warm-up images:", WARMUP_IMAGES)
print("CodeCarbon offline country:", COUNTRY_ISO_CODE)
print("Torch-Pruning available:", TORCH_PRUNING_AVAILABLE)


# ============================================================
# 3. File checks
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
# 4. Dataset
# ============================================================

test_df = pd.read_csv(TEST_CSV)

if SAMPLE_LIMIT is not None:
    test_df = test_df.head(SAMPLE_LIMIT).reset_index(drop=True)

label_map_df = pd.read_csv(LABEL_MAP_CSV).sort_values("label").reset_index(drop=True)
class_names = label_map_df["class_name"].tolist()
NUM_CLASSES = len(class_names)

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])


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


test_dataset = PlantVillageDataset(
    dataframe=test_df,
    dataset_root=DATASET_PATH,
    transform=test_transform,
)

eval_loader = DataLoader(
    test_dataset,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)

measure_loader = DataLoader(
    test_dataset,
    batch_size=MEASURE_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)

print("\nDataset information:")
print("Test images:", len(test_dataset))
print("Number of classes:", NUM_CLASSES)
print("Evaluation batches:", len(eval_loader))
print("Measurement batches:", len(measure_loader))


# ============================================================
# 5. Load full advanced model
# ============================================================

def load_full_model(model_path):
    loaded = torch.load(model_path, map_location=DEVICE, weights_only=False)

    if isinstance(loaded, torch.nn.Module):
        return loaded

    if isinstance(loaded, dict) and "model" in loaded:
        return loaded["model"]

    raise RuntimeError(f"Unsupported advanced model format: {type(loaded)}")


model = load_full_model(MODEL_PATH)
model = model.to(DEVICE)
model.eval()

print("\nModel loaded successfully.")


# ============================================================
# 6. Complexity utilities
# ============================================================

def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters()))


def get_model_size_mb(model_path):
    return float(Path(model_path).stat().st_size / (1024 * 1024))


def compute_or_get_macs(model):
    model.eval()

    if TORCH_PRUNING_AVAILABLE:
        try:
            example_inputs = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
            macs, params = tp.utils.count_ops_and_params(model, example_inputs)
            return float(macs), "computed_by_torch_pruning"
        except Exception as e:
            print("MAC computation failed:", repr(e))

    return float(KNOWN_MACS), "known_verified_value"


total_parameters = count_parameters(model)
model_size_mb = get_model_size_mb(MODEL_PATH)
macs, mac_source = compute_or_get_macs(model)

print("Model size MB:", model_size_mb)
print("Total parameters:", total_parameters)
print("MACs:", macs)
print("MAC source:", mac_source)


# ============================================================
# 7. Classification evaluation
# ============================================================

print("\nStarting classification evaluation...")

all_preds = []
all_labels = []

classification_start = time.perf_counter()

with torch.inference_mode():
    for images, labels in tqdm(eval_loader, desc="Advanced model classification evaluation"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

classification_end = time.perf_counter()
classification_time_seconds = classification_end - classification_start

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

accuracy = accuracy_score(all_labels, all_preds)

precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    all_labels,
    all_preds,
    average="macro",
    zero_division=0,
)

precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
    all_labels,
    all_preds,
    average="weighted",
    zero_division=0,
)

report_dict = classification_report(
    all_labels,
    all_preds,
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

cm = confusion_matrix(all_labels, all_preds)

classification_metrics = {
    "model_key": MODEL_KEY,
    "model_name": MODEL_NAME,
    "model_path": str(MODEL_PATH),
    "test_images": len(all_labels),
    "eval_batch_size": EVAL_BATCH_SIZE,
    "image_size": IMG_SIZE,
    "cpu_threads": torch.get_num_threads(),
    "model_size_mb": model_size_mb,
    "total_parameters": total_parameters,
    "parameters_million": total_parameters / 1_000_000,
    "macs": macs,
    "macs_million": macs / 1_000_000,
    "mac_source": mac_source,
    "test_accuracy": accuracy,
    "test_accuracy_percent": accuracy * 100,
    "macro_precision": precision_macro,
    "macro_precision_percent": precision_macro * 100,
    "macro_recall": recall_macro,
    "macro_recall_percent": recall_macro * 100,
    "macro_f1": f1_macro,
    "macro_f1_percent": f1_macro * 100,
    "weighted_precision": precision_weighted,
    "weighted_precision_percent": precision_weighted * 100,
    "weighted_recall": recall_weighted,
    "weighted_recall_percent": recall_weighted * 100,
    "weighted_f1": f1_weighted,
    "weighted_f1_percent": f1_weighted * 100,
    "classification_eval_time_seconds": classification_time_seconds,
    "avg_classification_eval_time_seconds_per_image": classification_time_seconds / len(all_labels),
}

pd.DataFrame([classification_metrics]).to_csv(
    METRICS_DIR / "advanced_pruned_classification_metrics.csv",
    index=False,
)

pd.DataFrame(report_dict).transpose().to_csv(
    METRICS_DIR / "advanced_pruned_classification_report.csv"
)

pd.DataFrame(cm).to_csv(
    METRICS_DIR / "advanced_pruned_confusion_matrix.csv",
    index=False,
)

pd.DataFrame({
    "true_label": all_labels,
    "predicted_label": all_preds,
}).to_csv(
    METRICS_DIR / "advanced_pruned_predictions.csv",
    index=False,
)

print("Classification accuracy %:", accuracy * 100)
print("Macro F1 %:", f1_macro * 100)
print("Weighted F1 %:", f1_weighted * 100)


# ============================================================
# 8. Latency, energy, CO2 measurement
# ============================================================

print("\nRunning warm-up inference...")

warmup_count = 0

with torch.inference_mode():
    for images, labels in measure_loader:
        images = images.to(DEVICE)
        _ = model(images)

        warmup_count += images.size(0)

        if warmup_count >= WARMUP_IMAGES:
            break

print("Warm-up completed. Images used:", warmup_count)

codecarbon_file = "advanced_pruned_codecarbon_bgd_offline.csv"

tracker = OfflineEmissionsTracker(
    output_dir=str(LATENCY_DIR),
    output_file=codecarbon_file,
    project_name="advanced_pruned_final_cpu_inference_bgd_offline",
    country_iso_code=COUNTRY_ISO_CODE,
    log_level="error",
)

all_forward_latencies = []
total_measured_images = 0

print("\nStarting measured CPU inference...")
tracker.start()

pipeline_start = time.perf_counter()

with torch.inference_mode():
    for images, labels in tqdm(measure_loader, desc="Advanced model latency-energy measurement"):
        images = images.to(DEVICE)

        forward_start = time.perf_counter()
        _ = model(images)
        forward_end = time.perf_counter()

        latency = (forward_end - forward_start) / images.size(0)

        all_forward_latencies.extend([latency] * images.size(0))
        total_measured_images += images.size(0)

pipeline_end = time.perf_counter()

emissions_kg_from_tracker = tracker.stop()

total_pipeline_time_seconds = pipeline_end - pipeline_start

latencies = np.array(all_forward_latencies)

avg_forward_latency_seconds = float(np.mean(latencies))
median_forward_latency_seconds = float(np.median(latencies))
min_forward_latency_seconds = float(np.min(latencies))
max_forward_latency_seconds = float(np.max(latencies))
std_forward_latency_seconds = float(np.std(latencies))

codecarbon_csv = LATENCY_DIR / codecarbon_file

energy_kwh = None
emissions_kg = emissions_kg_from_tracker
cpu_power = None
cpu_energy = None
ram_energy = None
recorded_country = COUNTRY_ISO_CODE

if codecarbon_csv.exists():
    emissions_df = pd.read_csv(codecarbon_csv)

    if len(emissions_df) > 0:
        last = emissions_df.iloc[-1]

        if "energy_consumed" in emissions_df.columns:
            energy_kwh = float(last["energy_consumed"])

        if "emissions" in emissions_df.columns:
            emissions_kg = float(last["emissions"])

        if "cpu_power" in emissions_df.columns:
            cpu_power = float(last["cpu_power"])

        if "cpu_energy" in emissions_df.columns:
            cpu_energy = float(last["cpu_energy"])

        if "ram_energy" in emissions_df.columns:
            ram_energy = float(last["ram_energy"])

        if "country_iso_code" in emissions_df.columns:
            recorded_country = str(last["country_iso_code"])

energy_per_image_kwh = energy_kwh / total_measured_images if energy_kwh is not None else None
co2_per_image_kg = emissions_kg / total_measured_images if emissions_kg is not None else None

latency_energy_co2_metrics = {
    "model_key": MODEL_KEY,
    "model_name": MODEL_NAME,
    "model_path": str(MODEL_PATH),
    "measurement_device": "Dell Latitude CPU",
    "measurement_process": "final_matched_cpu_measurement_bgd_offline",
    "country_iso_code_for_co2": COUNTRY_ISO_CODE,
    "country_iso_code_recorded": recorded_country,
    "total_images": total_measured_images,
    "measure_batch_size": MEASURE_BATCH_SIZE,
    "image_size": IMG_SIZE,
    "warmup_images": WARMUP_IMAGES,
    "cpu_threads": torch.get_num_threads(),
    "total_pipeline_time_seconds": total_pipeline_time_seconds,
    "avg_pipeline_time_seconds_per_image": total_pipeline_time_seconds / total_measured_images,
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
    "cpu_power_watt": cpu_power,
    "cpu_energy_kwh": cpu_energy,
    "ram_energy_kwh": ram_energy,
    "codecarbon_csv": str(codecarbon_csv),
    "measurement_note": (
        "Accuracy and F1 are not calculated inside this measured loop. "
        "This keeps latency-energy-CO2 measurement boundary clean and repeatable."
    ),
}

pd.DataFrame([latency_energy_co2_metrics]).to_csv(
    LATENCY_DIR / "advanced_pruned_latency_energy_co2_bgd_offline.csv",
    index=False,
)

pd.DataFrame({
    "image_index": list(range(total_measured_images)),
    "forward_latency_seconds": all_forward_latencies,
    "forward_latency_ms": latencies * 1000,
}).to_csv(
    LATENCY_DIR / "advanced_pruned_raw_forward_latency_values.csv",
    index=False,
)

print("Average forward latency ms/image:", avg_forward_latency_seconds * 1000)
print("Energy consumed kWh:", energy_kwh)
print("CO2 emissions kg:", emissions_kg)


# ============================================================
# 9. Figures
# ============================================================

def save_bar_figure(value, ylabel, title, filename, value_format="{:.2f}"):
    plt.figure(figsize=(6, 4))
    bars = plt.bar([MODEL_NAME], [value])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            value_format.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / f"{filename}.pdf", bbox_inches="tight")
    plt.close()


def save_confusion_matrix_figure(cm, filename):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Advanced Pruned Model Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / f"{filename}.pdf", bbox_inches="tight")
    plt.close()


def save_latency_histogram(latencies_ms, filename):
    plt.figure(figsize=(7, 4.5))
    plt.hist(latencies_ms, bins=40)
    plt.xlabel("Forward latency (ms/image)")
    plt.ylabel("Frequency")
    plt.title("Advanced Pruned Model Forward Latency Distribution")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / f"{filename}.pdf", bbox_inches="tight")
    plt.close()


def save_per_class_f1_figure(report_dict, filename):
    rows = []

    for class_name in class_names:
        if class_name in report_dict:
            rows.append({
                "class_name": class_name,
                "f1_score": report_dict[class_name]["f1-score"],
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("f1_score", ascending=True).head(15)

    plt.figure(figsize=(8, 6))
    plt.barh(df["class_name"], df["f1_score"] * 100)
    plt.xlabel("F1-score (%)")
    plt.title("Advanced Pruned Model Lowest 15 Per-class F1-scores")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / f"{filename}.pdf", bbox_inches="tight")
    plt.close()


save_bar_figure(
    accuracy * 100,
    "Accuracy (%)",
    "Advanced Pruned Model Test Accuracy",
    "advanced_pruned_test_accuracy",
    "{:.2f}",
)

save_bar_figure(
    f1_macro * 100,
    "Macro F1-score (%)",
    "Advanced Pruned Model Macro F1-score",
    "advanced_pruned_macro_f1",
    "{:.2f}",
)

save_bar_figure(
    model_size_mb,
    "Saved model size (MB)",
    "Advanced Pruned Model Size",
    "advanced_pruned_model_size",
    "{:.2f}",
)

save_bar_figure(
    total_parameters / 1_000_000,
    "Parameters (million)",
    "Advanced Pruned Parameter Count",
    "advanced_pruned_parameters",
    "{:.2f}",
)

save_bar_figure(
    macs / 1_000_000,
    "MACs (million)",
    "Advanced Pruned MACs",
    "advanced_pruned_macs",
    "{:.2f}",
)

save_bar_figure(
    avg_forward_latency_seconds * 1000,
    "Forward latency (ms/image)",
    "Advanced Pruned CPU Forward Latency",
    "advanced_pruned_forward_latency",
    "{:.2f}",
)

save_bar_figure(
    energy_kwh,
    "Total energy (kWh)",
    "Advanced Pruned Total Inference Energy",
    "advanced_pruned_total_energy",
    "{:.6f}",
)

save_bar_figure(
    emissions_kg,
    "CO2 emission (kgCO2eq)",
    "Advanced Pruned Estimated CO2 Emission",
    "advanced_pruned_co2_emission",
    "{:.8f}",
)

save_confusion_matrix_figure(cm, "advanced_pruned_confusion_matrix")
save_latency_histogram(latencies * 1000, "advanced_pruned_latency_histogram")
save_per_class_f1_figure(report_dict, "advanced_pruned_lowest_15_class_f1")


# ============================================================
# 10. Final combined files and summary
# ============================================================

combined_metrics = {
    **classification_metrics,
    **latency_energy_co2_metrics,
}

pd.DataFrame([combined_metrics]).to_csv(
    RESULT_ROOT / "advanced_pruned_final_all_metrics.csv",
    index=False,
)

environment_info = {
    "run_id": RUN_ID,
    "project_root": str(PROJECT_ROOT),
    "result_root": str(RESULT_ROOT),
    "model_path": str(MODEL_PATH),
    "dataset_path": str(DATASET_PATH),
    "test_csv": str(TEST_CSV),
    "test_images": len(test_dataset),
    "device": str(DEVICE),
    "torch_version": torch.__version__,
    "python_version": platform.python_version(),
    "os": platform.platform(),
    "cpu_count": os.cpu_count(),
    "cpu_threads": torch.get_num_threads(),
    "eval_batch_size": EVAL_BATCH_SIZE,
    "measure_batch_size": MEASURE_BATCH_SIZE,
    "warmup_images": WARMUP_IMAGES,
    "country_iso_code": COUNTRY_ISO_CODE,
}

with open(RESULT_ROOT / "advanced_pruned_run_environment_info.json", "w", encoding="utf-8") as f:
    json.dump(environment_info, f, indent=2)

with open(RESULT_ROOT / "advanced_pruned_final_key_findings.txt", "w", encoding="utf-8") as f:
    f.write("Final Advanced Safe Pointwise Incremental-Pruned FP32 MobileNetV2 Measurement\n")
    f.write("=" * 90 + "\n")
    f.write(f"Test accuracy: {accuracy * 100:.4f}%\n")
    f.write(f"Macro F1: {f1_macro * 100:.4f}%\n")
    f.write(f"Weighted F1: {f1_weighted * 100:.4f}%\n")
    f.write(f"Model size: {model_size_mb:.4f} MB\n")
    f.write(f"Parameters: {total_parameters:,}\n")
    f.write(f"MACs: {macs:,.0f}\n")
    f.write(f"Average forward latency: {avg_forward_latency_seconds * 1000:.4f} ms/image\n")
    f.write(f"Energy consumed: {energy_kwh:.12f} kWh\n")
    f.write(f"Energy per image: {energy_per_image_kwh:.12e} kWh/image\n")
    f.write(f"CO2 emissions: {emissions_kg:.12f} kgCO2eq\n")
    f.write(f"CO2 per image: {co2_per_image_kg:.12e} kgCO2eq/image\n")
    f.write("\nMeasurement protocol:\n")
    f.write("- Dell Latitude CPU\n")
    f.write("- Held-out test set\n")
    f.write("- Batch size 1 for latency-energy-CO2\n")
    f.write("- 50 warm-up images\n")
    f.write("- Offline CodeCarbon country_iso_code=BGD\n")
    f.write("\nModel description:\n")
    f.write("- Selected 11% structured-pruned model + additional 2% safe pointwise incremental pruning\n")
    f.write("- Do not report this as simple 13% pruning\n")
    f.write("- Report actual parameter, MAC, model-size, latency, energy, and CO2 reductions compared with baseline\n")

print("\n" + "=" * 100)
print("ADVANCED PRUNED FINAL MEASUREMENT COMPLETED")
print("=" * 100)
print("Result root:")
print(RESULT_ROOT)
print("\nMain file:")
print(RESULT_ROOT / "advanced_pruned_final_all_metrics.csv")
print("\nFigures folder:")
print(FIGURES_DIR)
print("=" * 100)