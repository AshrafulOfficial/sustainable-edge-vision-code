# ============================================================
# evaluate_baseline_cpu.py
# Baseline MobileNetV2 CPU Evaluation
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
BATCH_SIZE = 32
NUM_WORKERS = 0

DEVICE = torch.device("cpu")

# Keep CPU usage controlled
CPU_THREADS = max(1, os.cpu_count() - 1 if os.cpu_count() else 1)
torch.set_num_threads(CPU_THREADS)

print("=" * 70)
print("Baseline MobileNetV2 CPU Evaluation")
print("=" * 70)
print("Project root:", PROJECT_ROOT)
print("Dataset path:", DATASET_PATH)
print("Test CSV:", TEST_CSV)
print("Label map CSV:", LABEL_MAP_CSV)
print("Best model path:", BEST_MODEL_PATH)
print("Device:", DEVICE)
print("PyTorch version:", torch.__version__)
print("CPU threads:", torch.get_num_threads())


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

label_map_df = label_map_df.sort_values("label").reset_index(drop=True)
class_names = label_map_df["class_name"].tolist()
num_classes = len(class_names)

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

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
model_size_mb = BEST_MODEL_PATH.stat().st_size / (1024 * 1024)

print("\nModel loaded successfully.")
print("Checkpoint epoch:", checkpoint.get("epoch"))
print("Best validation accuracy:", checkpoint.get("best_val_accuracy"))
print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)
print("Model file size MB:", model_size_mb)
print("Classifier:", model.classifier)


# ============================================================
# 8. CPU Test Evaluation
# ============================================================

all_preds = []
all_labels = []

start_time = time.time()

with torch.inference_mode():
    for images, labels in tqdm(test_loader, desc="Evaluating on CPU"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

total_time = time.time() - start_time

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)


# ============================================================
# 9. Metrics
# ============================================================

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

avg_time_per_image = total_time / len(all_labels)

print("\n" + "=" * 70)
print("Baseline CPU Evaluation Results")
print("=" * 70)
print("Test images:", len(all_labels))
print("Total CPU evaluation time seconds:", total_time)
print("Average time per image seconds:", avg_time_per_image)
print("Accuracy:", accuracy)
print("Macro Precision:", precision_macro)
print("Macro Recall:", recall_macro)
print("Macro F1-score:", f1_macro)
print("Weighted Precision:", precision_weighted)
print("Weighted Recall:", recall_weighted)
print("Weighted F1-score:", f1_weighted)


# ============================================================
# 10. Save Metrics
# ============================================================

metrics = {
    "model_name": "MobileNetV2 Baseline",
    "evaluation_device": "Local CPU",
    "checkpoint_epoch": checkpoint.get("epoch"),
    "best_validation_accuracy": checkpoint.get("best_val_accuracy"),
    "test_images": len(all_labels),
    "test_accuracy": accuracy,
    "test_precision_macro": precision_macro,
    "test_recall_macro": recall_macro,
    "test_f1_macro": f1_macro,
    "test_precision_weighted": precision_weighted,
    "test_recall_weighted": recall_weighted,
    "test_f1_weighted": f1_weighted,
    "total_parameters": total_params,
    "trainable_parameters": trainable_params,
    "model_size_mb": model_size_mb,
    "total_cpu_evaluation_time_seconds": total_time,
    "average_time_per_image_seconds": avg_time_per_image,
    "image_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "cpu_threads": torch.get_num_threads(),
    "note": "This is local CPU batch evaluation time. Dedicated latency and energy measurement is performed separately.",
}

metrics_df = pd.DataFrame([metrics])

metrics_path = RESULT_DIR / "baseline_cpu_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)

print("\nCPU metrics saved at:")
print(metrics_path)


# ============================================================
# 11. Save Classification Report
# ============================================================

report_dict = classification_report(
    all_labels,
    all_preds,
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report_dict).transpose()

report_path = RESULT_DIR / "baseline_cpu_classification_report.csv"
report_df.to_csv(report_path)

print("\nCPU classification report saved at:")
print(report_path)


# ============================================================
# 12. Save Confusion Matrix
# ============================================================

cm = confusion_matrix(all_labels, all_preds)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names,
)

cm_csv_path = RESULT_DIR / "baseline_cpu_confusion_matrix.csv"
cm_df.to_csv(cm_csv_path)

print("\nCPU confusion matrix CSV saved at:")
print(cm_csv_path)

plt.figure(figsize=(18, 16))
plt.imshow(cm, interpolation="nearest")
plt.title("Baseline MobileNetV2 CPU Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=90, fontsize=7)
plt.yticks(tick_marks, class_names, fontsize=7)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

cm_png_path = RESULT_DIR / "baseline_cpu_confusion_matrix.png"
plt.savefig(cm_png_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nCPU confusion matrix image saved at:")
print(cm_png_path)

print("\nBaseline CPU evaluation completed successfully.")