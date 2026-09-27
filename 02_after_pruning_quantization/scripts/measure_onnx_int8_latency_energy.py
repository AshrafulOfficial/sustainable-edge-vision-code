# measure_onnx_int8_latency_energy.py
# ONNX INT8 model - local CPU latency, energy, CO2 measurement

import time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from codecarbon import EmissionsTracker
import onnxruntime as ort

# ── Paths ──
PROJECT_DIR    = Path(__file__).resolve().parents[2]
MODEL_PATH     = PROJECT_DIR / "02_after_pruning_quantization/models/mobilenetv2_pruned_11percent_onnx_int8.onnx"
TEST_CSV       = PROJECT_DIR / "data/test_files.csv"
DATASET_ROOT   = PROJECT_DIR / "data/PlantVillage"
RESULTS_DIR    = PROJECT_DIR / "02_after_pruning_quantization/results/onnx_int8_cpu"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE     = 224
BATCH_SIZE   = 1
WARMUP_IMAGES = 50

# ── Transform ──
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Dataset ──
class PlantVillageCSVDataset(Dataset):
    def __init__(self, csv_path, root, transform):
        self.df        = pd.read_csv(csv_path)
        self.root      = Path(root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.root / row["relative_path"]).convert("RGB")
        return self.transform(img), int(row["label"])

test_dataset = PlantVillageCSVDataset(TEST_CSV, DATASET_ROOT, transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Test images         : {len(test_dataset)}")
print(f"Model path exists   : {MODEL_PATH.exists()}")

# ── Load ONNX INT8 model ──
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 7

sess = ort.InferenceSession(
    str(MODEL_PATH),
    sess_options=sess_options,
    providers=["CPUExecutionProvider"]
)
input_name = sess.get_inputs()[0].name
print(f"Model loaded        : OK")
print(f"Input name          : {input_name}")

# ── Model size ──
model_size_mb = MODEL_PATH.stat().st_size / (1024 ** 2)
print(f"Model size          : {model_size_mb:.4f} MB")

# ── Warm-up ──
print(f"\nWarm-up ({WARMUP_IMAGES} images)...")
with_count = 0
for images, _ in test_loader:
    if with_count >= WARMUP_IMAGES:
        break
    sess.run(None, {input_name: images.numpy()})
    with_count += 1
print("Warm-up done.")

# ── Measurement ──
print("\nStarting measurement...")
latencies  = []
all_preds  = []
all_labels = []

tracker = EmissionsTracker(
    project_name   = "onnx_int8_cpu_inference",
    output_dir     = str(RESULTS_DIR),
    output_file    = "onnx_int8_codecarbon_emissions.csv",
    log_level      = "error"
)
tracker.start()
pipeline_start = time.perf_counter()

for images, labels in test_loader:
    img_np = images.numpy()
    start  = time.perf_counter()
    output = sess.run(None, {input_name: img_np})
    end    = time.perf_counter()

    latencies.append((end - start) * 1000)  # ms
    pred = int(np.argmax(output[0], axis=1)[0])
    all_preds.append(pred)
    all_labels.append(int(labels[0]))

emissions_kg   = tracker.stop()
pipeline_end   = time.perf_counter()
pipeline_time  = pipeline_end - pipeline_start

# ── Metrics ──
acc  = accuracy_score(all_labels, all_preds)
prec, rec, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="macro", zero_division=0
)
_, _, wf1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="weighted", zero_division=0
)

latencies_arr  = np.array(latencies)
avg_latency    = float(np.mean(latencies_arr))
median_latency = float(np.median(latencies_arr))
min_latency    = float(np.min(latencies_arr))
max_latency    = float(np.max(latencies_arr))
std_latency    = float(np.std(latencies_arr))

total_images   = len(all_labels)
avg_pipeline   = pipeline_time / total_images

# Energy
energy_kwh     = tracker._total_energy.kWh if hasattr(tracker, "_total_energy") else None
if energy_kwh is None or energy_kwh == 0:
    # codecarbon CSV থেকে পড়ো
    try:
        em_df      = pd.read_csv(RESULTS_DIR / "onnx_int8_codecarbon_emissions.csv")
        energy_kwh = float(em_df["energy_consumed"].iloc[-1])
    except Exception:
        energy_kwh = 0.0

energy_per_image = energy_kwh / total_images if total_images > 0 else 0
co2_kg           = emissions_kg if emissions_kg and emissions_kg > 0 else 0.0
co2_per_image    = co2_kg / total_images if total_images > 0 else 0

# ── Print results ──
print("\n" + "="*55)
print("ONNX INT8 CPU Measurement Results")
print("="*55)
print(f"Test images              : {total_images}")
print(f"Model size               : {model_size_mb:.4f} MB")
print(f"Accuracy                 : {acc:.6f} ({acc*100:.2f}%)")
print(f"Macro Precision          : {prec:.6f} ({prec*100:.2f}%)")
print(f"Macro Recall             : {rec:.6f} ({rec*100:.2f}%)")
print(f"Macro F1                 : {f1:.6f} ({f1*100:.2f}%)")
print(f"Weighted F1              : {wf1:.6f} ({wf1*100:.2f}%)")
print(f"Total pipeline time      : {pipeline_time:.4f} s")
print(f"Avg pipeline/image       : {avg_pipeline*1000:.4f} ms")
print(f"Avg forward latency      : {avg_latency:.4f} ms/image")
print(f"Median forward latency   : {median_latency:.4f} ms/image")
print(f"Min forward latency      : {min_latency:.4f} ms/image")
print(f"Max forward latency      : {max_latency:.4f} ms/image")
print(f"Std forward latency      : {std_latency:.4f} ms/image")
print(f"Energy consumed          : {energy_kwh:.10f} kWh")
print(f"Energy per image         : {energy_per_image:.10e} kWh/image")
print(f"CO2 emissions            : {co2_kg:.10f} kg")
print(f"CO2 per image            : {co2_per_image:.10e} kg/image")
print("="*55)

# ── Save results ──
results = {
    "model"                  : "ONNX INT8 Pruned MobileNetV2 11%",
    "test_images"            : total_images,
    "model_size_mb"          : round(model_size_mb, 6),
    "accuracy"               : round(acc,  6),
    "macro_precision"        : round(prec, 6),
    "macro_recall"           : round(rec,  6),
    "macro_f1"               : round(f1,   6),
    "weighted_f1"            : round(wf1,  6),
    "total_pipeline_time_s"  : round(pipeline_time, 6),
    "avg_pipeline_ms"        : round(avg_pipeline * 1000, 6),
    "avg_forward_latency_ms" : round(avg_latency,    6),
    "median_latency_ms"      : round(median_latency, 6),
    "min_latency_ms"         : round(min_latency,    6),
    "max_latency_ms"         : round(max_latency,    6),
    "std_latency_ms"         : round(std_latency,    6),
    "energy_kwh"             : energy_kwh,
    "energy_per_image_kwh"   : energy_per_image,
    "co2_kg"                 : co2_kg,
    "co2_per_image_kg"       : co2_per_image,
    "batch_size"             : BATCH_SIZE,
    "warmup_images"          : WARMUP_IMAGES,
    "cpu_threads"            : 7,
}

out_path = RESULTS_DIR / "onnx_int8_latency_energy.csv"
pd.DataFrame([results]).to_csv(out_path, index=False)
print(f"\nResults saved: {out_path}")