# ============================================================
# evaluate_pruned11_test_accuracy.py
#
# Purpose:
#   Compute the REAL held-out TEST-SET accuracy, macro-F1 and
#   weighted-F1 of the 11% Structured-Pruned FP32 MobileNetV2 model.
#
#   This number does NOT exist anywhere in the project yet — only
#   its VALIDATION accuracy (98.6251%) was recorded during pruning
#   candidate search. This script fills that gap so it can be
#   compared apples-to-apples against the Safe Pointwise+2%+KD
#   model's test accuracy (99.4292%).
#
# What it does:
#   1. Loads the 5,431-image held-out test set (same CSV/images
#      used everywhere else in the thesis).
#   2. Loads the 11% structured-pruned FULL MODEL object (not a
#      state_dict, because pruning changed the layer structure).
#   3. Runs inference on the test set (batch size 32, CPU).
#   4. Prints and saves Test Accuracy, Macro F1, Weighted F1,
#      a full classification report, and a confusion matrix.
#
# WHERE TO PUT THIS FILE:
#   thesis_project/02_after_pruning_quantization/scripts/evaluate_pruned11_test_accuracy.py
#
# WHERE TO RUN IT FROM:
#   cd into that exact "scripts" folder, then run it from there.
#   (See the command block at the very bottom of this file.)
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

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt


# ============================================================
# 1. Path Setup
# ============================================================
# This file lives at: thesis_project/02_after_pruning_quantization/scripts/
# parents[0] = scripts/
# parents[1] = 02_after_pruning_quantization/
# parents[2] = thesis_project/                     <-- PROJECT_ROOT

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "PlantVillage"

TEST_CSV = DATA_DIR / "test_files.csv"
LABEL_MAP_CSV = DATA_DIR / "label_map.csv"

OPT_DIR = PROJECT_ROOT / "02_after_pruning_quantization"
MODEL_DIR = OPT_DIR / "models"

RESULT_DIR = OPT_DIR / "results" / "pruned_11percent_real_test_evaluation"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Same fallback logic used elsewhere in your project scripts
PRUNED_MODEL_PATH = MODEL_DIR / "mobilenetv2_pruned_best_full_model.pth"
BACKUP_PRUNED_MODEL_PATH = MODEL_DIR / "mobilenetv2_safe_structured_pruned_11_percent_full_model.pth"

if not PRUNED_MODEL_PATH.exists() and BACKUP_PRUNED_MODEL_PATH.exists():
    PRUNED_MODEL_PATH = BACKUP_PRUNED_MODEL_PATH


# ============================================================
# 2. Configuration
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 32          # accuracy/F1 run -> batch 32 is fine (not a latency measurement)
NUM_WORKERS = 0
DEVICE = torch.device("cpu")

CPU_THREADS = max(1, os.cpu_count() - 1 if os.cpu_count() else 1)
torch.set_num_threads(CPU_THREADS)

print("=" * 70)
print("11% Structured-Pruned FP32 MobileNetV2 - REAL Test-Set Evaluation")
print("=" * 70)
print("Project root:      ", PROJECT_ROOT)
print("Dataset path:       ", DATASET_PATH)
print("Test CSV:           ", TEST_CSV)
print("Label map CSV:      ", LABEL_MAP_CSV)
print("Pruned model path:  ", PRUNED_MODEL_PATH)
print("Result folder:      ", RESULT_DIR)
print("Device:             ", DEVICE)
print("PyTorch version:    ", torch.__version__)
print("CPU threads:        ", torch.get_num_threads())


# ============================================================
# 3. File Checks
# ============================================================

required_paths = [DATASET_PATH, TEST_CSV, LABEL_MAP_CSV, PRUNED_MODEL_PATH]
for p in required_paths:
    if not p.exists():
        raise FileNotFoundError(f"Missing required path: {p}")

print("\nAll required files/folders found.")


# ============================================================
# 4. Load CSVs
# ============================================================

test_df = pd.read_csv(TEST_CSV)
label_map_df = pd.read_csv(LABEL_MAP_CSV).sort_values("label").reset_index(drop=True)
class_names = label_map_df["class_name"].tolist()
num_classes = len(class_names)

print("\nDataset information:")
print("Test images:        ", len(test_df))
print("Number of classes:  ", num_classes)


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

test_dataset = PlantVillageDataset(test_df, DATASET_PATH, test_transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)

print("\nDataLoader information:")
print("Test dataset size:  ", len(test_dataset))
print("Test batches:       ", len(test_loader))


# ============================================================
# 7. Load the 11% Pruned FULL MODEL
# ============================================================
# NOTE: Structured pruning physically removed channels/filters, so
# the saved file is a full nn.Module object (torch.save(model)),
# NOT a state_dict. Loading it into a fresh mobilenet_v2() would fail
# with a shape mismatch, so we load the object directly.

print("\nLoading 11% structured-pruned FULL model object...")

try:
    loaded_obj = torch.load(PRUNED_MODEL_PATH, map_location=DEVICE, weights_only=False)
except TypeError:
    loaded_obj = torch.load(PRUNED_MODEL_PATH, map_location=DEVICE)

if isinstance(loaded_obj, torch.nn.Module):
    model = loaded_obj
elif isinstance(loaded_obj, dict) and "model" in loaded_obj:
    model = loaded_obj["model"]
elif isinstance(loaded_obj, dict) and "model_state_dict" in loaded_obj:
    raise RuntimeError(
        "This file contains only model_state_dict. A structurally pruned "
        "model needs the full-model .pth file, e.g. "
        "mobilenetv2_pruned_best_full_model.pth or "
        "mobilenetv2_safe_structured_pruned_11_percent_full_model.pth."
    )
else:
    raise RuntimeError(f"Unsupported model file format: {type(loaded_obj)}")

model = model.to(DEVICE)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
model_size_mb = PRUNED_MODEL_PATH.stat().st_size / (1024 * 1024)

print("Model loaded successfully.")
print("Total parameters:   ", total_params)
print("Model file size MB: ", round(model_size_mb, 4))


# ============================================================
# 8. Run Test-Set Inference
# ============================================================

all_preds, all_labels = [], []
start_time = time.time()

with torch.inference_mode():
    for images, labels in tqdm(test_loader, desc="Evaluating 11% pruned model on TEST set"):
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

total_time = time.time() - start_time
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)


# ============================================================
# 9. Compute Metrics
# ============================================================

accuracy = accuracy_score(all_labels, all_preds)
p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="macro", zero_division=0)
p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="weighted", zero_division=0)

print("\n" + "=" * 70)
print("REAL TEST-SET RESULTS -- 11% Structured-Pruned FP32 MobileNetV2")
print("=" * 70)
print("Test images:            ", len(all_labels))
print("Test Accuracy:          ", round(accuracy * 100, 4), "%")
print("Test Macro Precision:   ", round(p_macro * 100, 4), "%")
print("Test Macro Recall:      ", round(r_macro * 100, 4), "%")
print("Test Macro F1:          ", round(f1_macro * 100, 4), "%")
print("Test Weighted F1:       ", round(f1_weighted * 100, 4), "%")
print("Total eval time (sec):  ", round(total_time, 2))


# ============================================================
# 10. Save Metrics CSV
# ============================================================

metrics = {
    "model_name": "11% Structured-Pruned FP32 MobileNetV2 (real test-set evaluation)",
    "model_path": str(PRUNED_MODEL_PATH),
    "test_images": len(all_labels),
    "test_accuracy": accuracy,
    "test_accuracy_percent": accuracy * 100,
    "test_precision_macro": p_macro,
    "test_recall_macro": r_macro,
    "test_f1_macro": f1_macro,
    "test_f1_macro_percent": f1_macro * 100,
    "test_precision_weighted": p_weighted,
    "test_recall_weighted": r_weighted,
    "test_f1_weighted": f1_weighted,
    "test_f1_weighted_percent": f1_weighted * 100,
    "total_parameters": total_params,
    "model_size_mb": model_size_mb,
    "total_eval_time_seconds": total_time,
    "batch_size": BATCH_SIZE,
    "image_size": IMG_SIZE,
    "note": "Fills the missing test-set evaluation gap for the 11% structured-pruned "
            "candidate, for direct comparison against the Safe Pointwise+2%+KD test results.",
}

metrics_path = RESULT_DIR / "pruned11_real_test_metrics.csv"
pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
print("\nSaved metrics CSV:  ", metrics_path)


# ============================================================
# 11. Save Classification Report
# ============================================================

report_df = pd.DataFrame(
    classification_report(
        all_labels, all_preds, target_names=class_names,
        output_dict=True, zero_division=0,
    )
).transpose()

report_path = RESULT_DIR / "pruned11_real_test_classification_report.csv"
report_df.to_csv(report_path)
print("Saved classification report:", report_path)


# ============================================================
# 12. Save Confusion Matrix (CSV + PNG)
# ============================================================

cm = confusion_matrix(all_labels, all_preds)
cm_csv_path = RESULT_DIR / "pruned11_real_test_confusion_matrix.csv"
pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(cm_csv_path)
print("Saved confusion matrix CSV: ", cm_csv_path)

plt.figure(figsize=(18, 16))
plt.imshow(cm, interpolation="nearest")
plt.title("11% Structured-Pruned FP32 MobileNetV2 - Test Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=90, fontsize=7)
plt.yticks(tick_marks, class_names, fontsize=7)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

cm_png_path = RESULT_DIR / "pruned11_real_test_confusion_matrix.png"
plt.savefig(cm_png_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved confusion matrix PNG: ", cm_png_path)

print("\n" + "=" * 70)
print("DONE. All outputs saved in:")
print(RESULT_DIR)
print("=" * 70)


# ============================================================
# HOW TO USE THIS SCRIPT
# ============================================================
#
# 1) WHERE TO PUT THIS FILE:
#    Save/copy it into:
#    thesis_project/02_after_pruning_quantization/scripts/evaluate_pruned11_test_accuracy.py
#
# 2) WHERE TO RUN IT FROM:
#    Open your terminal (with the .venv activated), then:
#
#      cd "<PROJECT_ROOT>/02_after_pruning_quantization/scripts"
#      python3 evaluate_pruned11_test_accuracy.py
#
#    (If your venv is not active yet:
#      source "<PROJECT_ROOT>/.venv/bin/activate"
#    )
#
# 3) WHAT YOU GET:
#    Printed in terminal: Test Accuracy, Macro F1, Weighted F1
#    Saved to disk (new folder, nothing old is touched or overwritten):
#      02_after_pruning_quantization/results/pruned_11percent_real_test_evaluation/
#        - pruned11_real_test_metrics.csv
#        - pruned11_real_test_classification_report.csv
#        - pruned11_real_test_confusion_matrix.csv
#        - pruned11_real_test_confusion_matrix.png
# ============================================================