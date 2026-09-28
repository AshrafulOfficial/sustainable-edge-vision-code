# Sustainable Edge Vision — Code

Code for the paper:

> **Sustainable Edge Vision: An Energy-efficient Deep Learning for Crop Disease Detection**
> Md Ashraful Islam
> *IEEE i-COSTE 2026* (Paper ID: `ieee-icoste_4395`)

This repository contains the notebooks and scripts used to train, prune, distill, and measure a MobileNetV2 crop-disease classifier on the [PlantVillage](https://arxiv.org/abs/1604.03169) dataset (38 classes). Latency, energy, and estimated CO₂ were measured on a CPU-only laptop.

---

## Method at a glance

A two-stage compression pipeline applied to a MobileNetV2 baseline:

1. **Stage 1: structured channel pruning.** Filters are ranked by ℓ2-norm and removed with dependency-aware pruning ([Torch-Pruning](https://github.com/VainF/Torch-Pruning)). Ratios 5%–11% were searched and 11% was selected.
2. **Stage 2: safe pointwise incremental pruning.** An extra **β = 2%** of 1×1 pointwise channels is removed. Depthwise, stem, final feature, classifier, and narrow (< 64 channels) layers are protected. Widths are rounded to multiples of 8.
3. **Knowledge distillation.** The frozen baseline teaches the pruned student to recover accuracy.

| KD setting | Value |
|---|---|
| Mixing coefficient α | 0.5 |
| Temperature T | 4.0 |
| Fine-tuning epochs | 10 (best validation checkpoint kept) |
| Optimizer | AdamW, lr 3e-5, weight decay 1e-4, cosine schedule |
| Label smoothing | 0.05 |

Notation: **β** is the Stage-2 pruning ratio, **α** is the KD mixing coefficient.

## Results

Test set: 5,431 images. CPU inference, batch size 1.

| Metric | Baseline | Final (pruned + KD) | Change |
|---|---|---|---|
| Test accuracy | 99.72% | 99.43% | −0.29 pp |
| Macro F1 | 99.57% | 99.23% | −0.34 pp |
| Parameters | 2,272,550 | 1,737,890 | −23.53% |
| MACs | 319.00 M | 249.27 M | −21.86% |
| Model size (FP32 weights) | 8.90 MiB | 6.84 MiB | −23.14% |
| CPU latency | 34.76 ms | 25.59 ms | −26.39% |
| Inference energy | 0.001294 kWh | 0.000926 kWh | −28.43% |
| Estimated CO₂ | 0.000895 kg | 0.000640 kg | −28.43% |

Notes:

- **CO₂ is derived from energy.** CO₂ = energy × grid carbon intensity (about 0.6914 kg CO₂/kWh for Bangladesh in CodeCarbon), so the two reductions are the same number, not two independent findings.
- **Model size protocol.** Both sizes come from `torch.save(model.state_dict())` in FP32 with no optimizer state, produced by `03_final_comparison/scripts/recompute_model_size.py`. Outputs are in `03_final_comparison/results/model_size/`. An earlier comparison used a baseline training checkpoint that also stored Adam optimizer state, which overstated the size reduction; that figure was withdrawn.
- **Latency and energy** in the table come from a single full-test-set pass per model. For mean ± standard deviation over repeated runs, use `03_final_comparison/scripts/repeated_runs_latency_energy.py`.

## Scope and limitations

- Measurements are **CPU-simulated edge conditions** on a laptop-class CPU, not on physical edge hardware such as a Raspberry Pi or a phone.
- Energy comes from [CodeCarbon](https://github.com/mlco2/codecarbon), not an external power meter. Depending on the platform it reads Intel RAPL counters or falls back to a CPU-model-based estimate. The scripts used to produce the paper's numbers ran with `log_level="error"`, so the active power source was not recorded. The repeated-runs script logs it.
- Background processes and CPU frequency scaling were not explicitly controlled.
- Only PlantVillage (controlled-condition images) and MobileNetV2 were evaluated, with a single training seed.
- Quantization (INT8) experiments are included as exploratory work and are not part of the reported framework.

## Repository structure

```
.
├── 01_baseline_before_optimization/
│   ├── notebooks/   01_dataset_setup_and_split, 02_mobilenetv2_baseline_training, 03_baseline_evaluation_colab
│   └── scripts/     evaluate_baseline_cpu, measure_baseline_latency_energy, create_official_baseline_final_results
│
├── 02_after_pruning_quantization/
│   ├── notebooks/   04_pruning_finetuning_colab (Stage 1), 05_int8_quantization_colab
│   ├── scripts/     pruned-FP32 and ONNX-INT8 measurement scripts
│   └── advanced_pruning_experiments/
│       ├── notebooks/   07_advanced_energy_aware_pruning_experiment_colab (Stage 2 + KD)
│       └── scripts/     final CPU measurement of the advanced model
│
├── 03_final_comparison/
│   └── scripts/     final_compare_baseline_official_vs_advanced,
│                    recompute_model_size, repeated_runs_latency_energy
│
└── requirements.txt
```

Model checkpoints and the image data are not included. See below.

## Environment

| Component | Description |
|---|---|
| Device | Dell Latitude 7390 laptop, CPU only |
| CPU | Intel Core i5-8350U (4 cores / 8 threads, 1.70 GHz), 7 threads used |
| RAM | ≈ 7.63 GB |
| OS | Ubuntu 24.04 LTS |
| Framework | PyTorch 2.12.0 / TorchVision 0.27.0 (CPU build) |
| Language | Python 3.12.3 |
| Energy measurement | CodeCarbon 3.2.7 (country code BGD) |
| Input / batch | 224 × 224 RGB, batch size 1, 50 warm-up images |

## Setup

```bash
git clone https://github.com/AshrafulOfficial/sustainable-edge-vision-code.git
cd sustainable-edge-vision-code

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data and checkpoints

The scripts resolve paths relative to the repository root and expect:

```
data/
├── PlantVillage/          # image folders (public dataset)
├── train_files.csv
├── val_files.csv
├── test_files.csv
└── label_map.csv
```

Download PlantVillage, then run `01_dataset_setup_and_split.ipynb` to create the CSV splits (75% / 15% / 10%: 40,728 train, 8,146 validation, 5,431 test).

Trained checkpoints are not hosted here. The scripts expect them in each stage's `models/` folder (for example `01_baseline_before_optimization/models/mobilenetv2_baseline_best.pth`). You can regenerate them with the notebooks, or request the weights from the author.

## Reproducing the results

Run in this order:

1. `01_baseline_before_optimization/notebooks/01_dataset_setup_and_split.ipynb`: dataset splits
2. `.../02_mobilenetv2_baseline_training.ipynb`: baseline training
3. `.../03_baseline_evaluation_colab.ipynb` and `scripts/evaluate_baseline_cpu.py`: baseline accuracy
4. `02_after_pruning_quantization/notebooks/04_pruning_finetuning_colab.ipynb`: Stage 1 search (5%–11%)
5. `02_after_pruning_quantization/advanced_pruning_experiments/notebooks/07_advanced_energy_aware_pruning_experiment_colab.ipynb`: Stage 2 pruning and KD. The extra ratio is set by `EXTRA_PRUNING_RATIOS` (β = 0.02 for the reported model).
6. Measurement on your CPU:
   ```bash
   python 01_baseline_before_optimization/scripts/measure_baseline_latency_energy.py
   python 02_after_pruning_quantization/advanced_pruning_experiments/scripts/final_measure_advanced_pruned_cpu_bgd_offline.py
   ```
7. Comparison and tables:
   ```bash
   python 03_final_comparison/scripts/final_compare_baseline_official_vs_advanced.py
   ```
8. Model size under one protocol:
   ```bash
   python 03_final_comparison/scripts/recompute_model_size.py \
     --baseline-ckpt 01_baseline_before_optimization/models/mobilenetv2_baseline_best.pth \
     --final-model 02_after_pruning_quantization/advanced_pruning_experiments/models/<final_model>.pth
   ```
9. Optional, mean ± std over repeated runs (alternates models in one session, logs the CodeCarbon power source):
   ```bash
   python 03_final_comparison/scripts/repeated_runs_latency_energy.py --runs 5 | tee measurement_log.txt
   ```

Notebooks with the `_colab` suffix were run on Colab and expect the data to be available under the same relative layout.

## Citation

```bibtex
@inproceedings{islam2026sustainableedgevision,
  title     = {Sustainable Edge Vision: An Energy-efficient Deep Learning for Crop Disease Detection},
  author    = {Islam, Md Ashraful},
  booktitle = {Proceedings of IEEE i-COSTE 2026},
  year      = {2026}
}
```

The entry will be updated once the proceedings details are available.

## License

No license has been specified yet. Please contact the author before reusing the code beyond inspection.

## Contact

Md Ashraful Islam. Questions and checkpoint requests: please open a GitHub issue.