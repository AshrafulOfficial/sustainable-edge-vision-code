# ============================================================
# evaluate_safe_pointwise_kd_extra2_cpu.py
# Advanced Safe Pointwise KD Pruned Candidate
# Dell CPU Test Accuracy Evaluation
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

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


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

RESULT_DIR = RESULT_ROOT / "safe_pointwise_kd_extra2_cpu_test_evaluation"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Configuration
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0
DEVICE = torch.device("cpu")

CPU_THREADS = max(1, os.cpu_count() - 1 if os.cpu_count() else 1)
torch.set_num_threads(CPU_THREADS)

print("=" * 90)
print("Advanced Safe Pointwise KD Extra-2% Candidate CPU Test Evaluation")
print("=" * 90)
print("Project root:", PROJECT_ROOT)
print("Dataset path:", DATASET_PATH)
print("Test CSV:", TEST_CSV)
print("Label map CSV:", LABEL_MAP_CSV)
print("Model path:", MODEL_PATH)
print("Result folder:", RESULT_DIR)
print("Device:", DEVICE)
print("PyTorch version:", torch.__version__)
print("CPU threads:", torch.get_num_threads())
print("Batch size:", BATCH_SIZE)
print("Image size:", IMG_SIZE)


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
label_map_df = pd.read_csv(LABEL_MAP_CSV).sort_values("label").reset_index(drop=True)

num_classes = len(label_map_df)

print("\nDataset information:")
print("Test images:", len(test_df))
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
print("Test dataset size:", len(test_dataset))
print("Test batches:", len(test_loader))


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
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Model loaded successfully.")
print("Model size MB:", model_size_mb)
print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)


# ============================================================
# 8. Test Evaluation
# ============================================================

print("\nStarting CPU test evaluation...")

all_preds = []
all_labels = []

start_time = time.perf_counter()

with torch.inference_mode():
    for images, labels in tqdm(test_loader, desc="Evaluating advanced candidate on Dell CPU"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

end_time = time.perf_counter()

total_eval_time = end_time - start_time

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
    target_names=label_map_df["class_name"].tolist(),
    output_dict=True,
    zero_division=0,
)

cm = confusion_matrix(all_labels, all_preds)


# ============================================================
# 9. Print Results
# ============================================================

print("\n" + "=" * 90)
print("Advanced Candidate CPU Test Results")
print("=" * 90)
print("Test images:", len(all_labels))
print("Accuracy:", accuracy)
print("Accuracy %:", accuracy * 100)
print("Macro precision:", precision_macro)
print("Macro recall:", recall_macro)
print("Macro F1:", f1_macro)
print("Weighted precision:", precision_weighted)
print("Weighted recall:", recall_weighted)
print("Weighted F1:", f1_weighted)
print("Model size MB:", model_size_mb)
print("Total parameters:", total_params)
print("Total CPU evaluation time seconds:", total_eval_time)
print("Average evaluation time seconds/image:", total_eval_time / len(all_labels))


# ============================================================
# 10. Save Results
# ============================================================

metrics = {
    "model_name": "safe_pointwise_kd_from_11p_extra_2percent",
    "model_status": "advanced_pruning_candidate_NOT_FINAL",
    "evaluation_device": "Dell Latitude CPU",
    "model_path": str(MODEL_PATH),
    "test_images": len(all_labels),
    "batch_size": BATCH_SIZE,
    "image_size": IMG_SIZE,
    "cpu_threads": torch.get_num_threads(),
    "model_size_mb": model_size_mb,
    "total_parameters": total_params,
    "trainable_parameters": trainable_params,
    "test_accuracy": accuracy,
    "test_accuracy_percent": accuracy * 100,
    "macro_precision": precision_macro,
    "macro_recall": recall_macro,
    "macro_f1": f1_macro,
    "macro_f1_percent": f1_macro * 100,
    "weighted_precision": precision_weighted,
    "weighted_recall": recall_weighted,
    "weighted_f1": f1_weighted,
    "weighted_f1_percent": f1_weighted * 100,
    "total_cpu_evaluation_time_seconds": total_eval_time,
    "avg_evaluation_time_seconds_per_image": total_eval_time / len(all_labels),
    "note": (
        "CPU test classification evaluation for the advanced safe pointwise KD extra-2% "
        "candidate. This is accuracy/F1 evaluation only. Final latency-energy must be "
        "measured separately with batch size 1 and matched baseline process."
    ),
}

metrics_path = RESULT_DIR / "safe_pointwise_kd_extra2_cpu_test_metrics.csv"
pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

report_path = RESULT_DIR / "safe_pointwise_kd_extra2_cpu_classification_report.csv"
pd.DataFrame(report_dict).transpose().to_csv(report_path)

cm_path = RESULT_DIR / "safe_pointwise_kd_extra2_cpu_confusion_matrix.csv"
pd.DataFrame(cm).to_csv(cm_path, index=False)

preds_path = RESULT_DIR / "safe_pointwise_kd_extra2_cpu_predictions.csv"
pd.DataFrame({
    "true_label": all_labels,
    "predicted_label": all_preds,
}).to_csv(preds_path, index=False)

key_findings_path = RESULT_DIR / "safe_pointwise_kd_extra2_cpu_key_findings.txt"
with open(key_findings_path, "w", encoding="utf-8") as f:
    f.write("Advanced Safe Pointwise KD Extra-2% Candidate - CPU Test Evaluation\n")
    f.write("=" * 80 + "\n")
    f.write(f"Test accuracy: {accuracy * 100:.4f}%\n")
    f.write(f"Macro F1: {f1_macro * 100:.4f}%\n")
    f.write(f"Weighted F1: {f1_weighted * 100:.4f}%\n")
    f.write(f"Model size: {model_size_mb:.4f} MB\n")
    f.write(f"Parameters: {total_params:,}\n")
    f.write(f"Evaluation time: {total_eval_time:.4f} seconds\n")
    f.write("\nNote: This candidate is not final until matched latency-energy measurement is completed.\n")

print("\nSaved files:")
print(metrics_path)
print(report_path)
print(cm_path)
print(preds_path)
print(key_findings_path)

print("\nCPU test evaluation completed successfully.")