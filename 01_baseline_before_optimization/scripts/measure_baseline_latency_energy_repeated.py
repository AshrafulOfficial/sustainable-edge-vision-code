#!/usr/bin/env python3
"""
measure_baseline_latency_energy_repeated.py

10 repeated CPU latency / energy / CO2 measurements of the baseline model.
- Forces CodeCarbon's country to Bangladesh (BGD) explicitly via
  OfflineEmissionsTracker, so geolocation can NEVER drift to another country
  (this is what caused the earlier "Canada/Quebec" near-zero CO2 result --
  the original script used the online, IP-geolocated EmissionsTracker).
- Saves everything under:
    01_baseline_before_optimization/final_results/baseline_official_reference_final/
  with a new, clearly-dated name (raw per-run CSV + a summary with mean +/- std),
  so it does not collide with or silently overwrite the old hardcoded "official" file.

Run from the project root:
    python 01_baseline_before_optimization/scripts/measure_baseline_latency_energy_repeated.py
    python .../measure_baseline_latency_energy_repeated.py --runs 10 --cooldown 30

Before running:
    cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort -u   # must print "performance"
    cat /sys/class/power_supply/A*/online                                 # must print 1 (on AC power)
"""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from codecarbon import OfflineEmissionsTracker

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "PlantVillage"
TEST_CSV = DATA_DIR / "test_files.csv"
LABEL_MAP_CSV = DATA_DIR / "label_map.csv"

BASELINE_DIR = PROJECT_ROOT / "01_baseline_before_optimization"
BEST_MODEL_PATH = BASELINE_DIR / "models" / "mobilenetv2_baseline_best.pth"

OUT_DIR = BASELINE_DIR / "final_results" / "baseline_official_reference_final"
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUNS_SUBDIR = OUT_DIR / f"repeated_runs_{RUN_STAMP}"

IMG_SIZE = 224
BATCH_SIZE = 1
WARMUP_IMAGES = 50
COUNTRY_ISO_CODE = "BGD"  # Bangladesh -- forced, no IP geolocation


class PlantVillageDataset(Dataset):
    def __init__(self, dataframe, dataset_root, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = Image.open(self.dataset_root / row["relative_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"])


def build_loader(limit=None):
    test_df = pd.read_csv(TEST_CSV)
    label_map_df = pd.read_csv(LABEL_MAP_CSV).sort_values("label").reset_index(drop=True)
    if limit:
        test_df = test_df.head(limit).reset_index(drop=True)
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = PlantVillageDataset(test_df, DATASET_PATH, tf)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    return loader, len(label_map_df)


def load_model(num_classes):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    ckpt = torch.load(BEST_MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval(), ckpt


def run_once(model, loader, run_idx, out_dir):
    seen = 0
    with torch.inference_mode():
        for images, _ in loader:
            model(images)
            seen += images.size(0)
            if seen >= WARMUP_IMAGES:
                break

    csv_name = f"baseline_codecarbon_run{run_idx}.csv"
    tracker = OfflineEmissionsTracker(
        output_dir=str(out_dir),
        output_file=csv_name,
        project_name=f"MobileNetV2_Baseline_CPU_Inference_run{run_idx}",
        country_iso_code=COUNTRY_ISO_CODE,
        log_level="info",
    )

    latencies = []
    total_images = 0
    tracker.start()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for images, _ in tqdm(loader, desc=f"baseline run {run_idx}", leave=False):
            a = time.perf_counter()
            model(images)
            b = time.perf_counter()
            latencies.append((b - a) / images.size(0))
            total_images += images.size(0)
    wall = time.perf_counter() - t0
    tracker.stop()

    row = pd.read_csv(out_dir / csv_name).iloc[-1]
    lat_ms = np.array(latencies) * 1000
    return {
        "run": run_idx,
        "total_images": total_images,
        "total_pipeline_time_seconds": wall,
        "avg_forward_latency_ms_per_image": float(lat_ms.mean()),
        "median_forward_latency_ms_per_image": float(np.median(lat_ms)),
        "min_forward_latency_ms_per_image": float(lat_ms.min()),
        "max_forward_latency_ms_per_image": float(lat_ms.max()),
        "std_forward_latency_ms_per_image": float(lat_ms.std()),
        "energy_consumed_kwh": float(row["energy_consumed"]),
        "co2_emissions_kg": float(row["emissions"]),
        "cpu_power_w": float(row.get("cpu_power", float("nan"))),
        "country_iso_code": row.get("country_iso_code", ""),
        "cpu_utilization_percent": float(row.get("cpu_utilization_percent", float("nan"))),
    }


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), (float(arr.std(ddof=1)) if len(arr) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="quick test: use only first N images")
    ap.add_argument("--cooldown", type=int, default=30, help="seconds to idle between runs")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    RUNS_SUBDIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Baseline MobileNetV2 -- repeated CPU latency/energy/CO2 measurement")
    print("=" * 80)
    print("Project root:", PROJECT_ROOT)
    print("Output dir  :", RUNS_SUBDIR)
    print("Runs        :", args.runs)
    print("CPU threads :", torch.get_num_threads())
    print("Forced country (CodeCarbon):", COUNTRY_ISO_CODE, "(Dhaka, Bangladesh)")
    print("PyTorch     :", torch.__version__)
    print()
    print("Reminder -- before trusting these numbers, confirm:")
    print("  cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort -u   -> performance")
    print("  cat /sys/class/power_supply/A*/online                                -> 1")
    print()

    loader, num_classes = build_loader(args.limit)
    model, ckpt = load_model(num_classes)
    model_size_mb = BEST_MODEL_PATH.stat().st_size / (1024 * 1024)
    total_params = sum(p.numel() for p in model.parameters())

    rows = []
    for i in range(1, args.runs + 1):
        r = run_once(model, loader, i, RUNS_SUBDIR)
        rows.append(r)
        print(f"  run {i:2d}: latency {r['avg_forward_latency_ms_per_image']:.3f} ms | "
              f"energy {r['energy_consumed_kwh']:.7f} kWh | "
              f"CO2 {r['co2_emissions_kg']:.7f} kg | country {r['country_iso_code']}")
        pd.DataFrame(rows).to_csv(RUNS_SUBDIR / "baseline_repeated_runs_raw.csv", index=False)
        if i < args.runs:
            time.sleep(args.cooldown)

    df = pd.DataFrame(rows)
    if not (df["country_iso_code"] == COUNTRY_ISO_CODE).all():
        print("\n*** WARNING: not all runs report country_iso_code == BGD -- check the raw CSV. ***\n")

    summary = {
        "model_name": "MobileNetV2 Baseline",
        "evaluation_device": "Dell Latitude 7390 CPU",
        "checkpoint_epoch": ckpt.get("epoch"),
        "best_validation_accuracy": ckpt.get("best_val_accuracy"),
        "total_images": int(df["total_images"].iloc[0]),
        "batch_size": BATCH_SIZE,
        "image_size": IMG_SIZE,
        "cpu_threads": torch.get_num_threads(),
        "model_size_mb": model_size_mb,
        "total_parameters": total_params,
        "num_runs": args.runs,
        "country_iso_code": COUNTRY_ISO_CODE,
        "avg_forward_latency_ms_per_image_mean": mean_std(df.avg_forward_latency_ms_per_image)[0],
        "avg_forward_latency_ms_per_image_std": mean_std(df.avg_forward_latency_ms_per_image)[1],
        "energy_consumed_kwh_mean": mean_std(df.energy_consumed_kwh)[0],
        "energy_consumed_kwh_std": mean_std(df.energy_consumed_kwh)[1],
        "co2_emissions_kg_mean": mean_std(df.co2_emissions_kg)[0],
        "co2_emissions_kg_std": mean_std(df.co2_emissions_kg)[1],
        "total_pipeline_time_seconds_sum": float(df.total_pipeline_time_seconds.sum()),
        "note": ("Repeated (N=%d) baseline CPU inference measurement, performance governor, "
                 "CodeCarbon country forced to BGD (Dhaka). Supersedes the earlier single-run "
                 "'official' hardcoded values in create_official_baseline_final_results.py "
                 "and the stray Canada-geolocated run in results/baseline_latency_energy.csv." % args.runs),
    }
    (RUNS_SUBDIR / "baseline_repeated_runs_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame([summary]).to_csv(OUT_DIR / "baseline_official_latency_energy_co2_REPEATED.csv", index=False)

    print("\n" + "=" * 80)
    print(f"Latency : {summary['avg_forward_latency_ms_per_image_mean']:.3f} +/- "
          f"{summary['avg_forward_latency_ms_per_image_std']:.3f} ms")
    print(f"Energy  : {summary['energy_consumed_kwh_mean']:.7f} +/- "
          f"{summary['energy_consumed_kwh_std']:.7f} kWh")
    print(f"CO2     : {summary['co2_emissions_kg_mean']:.7f} +/- "
          f"{summary['co2_emissions_kg_std']:.7f} kg")
    print("\nSaved per-run data to:", RUNS_SUBDIR)
    print("Saved summary CSV to :", OUT_DIR / "baseline_official_latency_energy_co2_REPEATED.csv")
    print("Saved summary JSON to:", RUNS_SUBDIR / "baseline_repeated_runs_summary.json")


if __name__ == "__main__":
    main()