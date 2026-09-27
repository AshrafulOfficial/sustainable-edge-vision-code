# Sustainable Edge Vision — Code

Official code repository for:

> **Sustainable Edge Vision: An Energy-efficient Deep Learning for Crop Disease Detection**
> Md Ashraful Islam
> *Accepted at IEEE i-COSTE 2026* (Paper ID: `ieee-icoste_4395`)

This repository contains the training, pruning, knowledge-distillation, quantization, and CPU energy/latency measurement code used to produce the results reported in the paper. A reproducible **Green AI / Edge AI evaluation pipeline** for a MobileNetV2-based crop disease classifier, trained on the [PlantVillage](https://arxiv.org/abs/1604.03169) dataset (38 classes).

---

## Overview

Deploying CNNs on resource-limited edge devices is constrained by computational cost, memory, latency, and power consumption. This work applies a **two-stage compression framework** to a MobileNetV2 baseline:

1. **Structured channel pruning** (Taylor/ℓ2-norm–based), searched across 5–11% pruning ratios
2. **Safe pointwise incremental pruning** — a small, layer-protected additional pruning step on 1×1 pointwise convolutions
3. **Knowledge distillation** from the original (frozen) baseline to recover accuracy lost to compression

All models are evaluated on a **CPU-only** platform, with inference latency, energy consumption (via [CodeCarbon](https://github.com/mlco2/codecarbon)), and estimated CO₂ emissions measured directly — not simulated.

## Key Results

| Metric | Baseline | Final (Pruned + KD) | Change |
|---|---|---|---|
| Test accuracy | 99.72% | 99.43% | −0.29 pp |
| Macro F1 | 99.57% | 99.23% | −0.34 pp |
| Parameters | 2,272,550 | 1,737,890 | **−23.53%** |
| MACs | 319.00M | 249.27M | **−21.86%** |
| CPU latency (batch=1) | 34.76 ms | 25.59 ms | **−26.39%** |
| Inference energy | 0.001294 kWh | 0.000926 kWh | **−28.43%** |
| Estimated CO₂ | 0.000895 kg | 0.000640 kg | **−28.43%** |

> **Note on model size:** the on-disk model-size figure in the original submission included optimizer state in the baseline checkpoint, which is not an apples-to-apples comparison with the final model's clean weights-only checkpoint. The corrected comparison (clean weights-only for both models) is being reflected in the camera-ready version; see `01_baseline_before_optimization/scripts/create_official_baseline_final_results.py` for the size computation.

All measurements were taken under **CPU-simulated edge conditions** on a laptop-class CPU (see [Environment](#environment) below) — not on physical edge hardware (e.g., Raspberry Pi). Real-device validation is planned future work.

## Repository Structure

```
.
├── 01_baseline_before_optimization/
│   ├── notebooks/          # Dataset setup, baseline MobileNetV2 training, evaluation
│   └── scripts/            # Official baseline results generation, CPU latency/energy measurement
│
├── 02_after_pruning_quantization/
│   ├── notebooks/          # Structured pruning search + fine-tuning, INT8 quantization
│   ├── scripts/            # Pruned-model / ONNX-INT8 measurement scripts
│   ├── advanced_pruning_experiments/
│   │   ├── notebooks/      # Safe pointwise incremental pruning + knowledge distillation
│   │   └── scripts/        # Final advanced-model CPU measurement
│   └── quantization_experiments/   # QAT / PTQ INT8 experiment artifacts
│
├── 03_final_comparison/
│   ├── notebooks/
│   └── scripts/            # Baseline vs. final model comparison, paper tables/figures
│
└── requirements.txt
```

Trained model checkpoints (`.pth`) and the PlantVillage image data are **not** included in this repository to keep it lightweight — see [Data & Checkpoints](#data--checkpoints) below.

## Environment

| Component | Description |
|---|---|
| CPU | Intel Core i5-8350U (4 physical cores / 8 logical threads, 1.70 GHz) |
| RAM | ≈7.63 GB |
| OS | Ubuntu 24.04 LTS |
| Framework | PyTorch 2.12.0 / TorchVision 0.27.0 (CPU build) |
| Language | Python 3.12.3 |
| Energy measurement | CodeCarbon v3.2.7 |
| Input size | 224 × 224 RGB |
| Inference batch size | 1 |

## Setup

```bash
git clone https://github.com/AshrafulOfficial/sustainable-edge-vision-code.git
cd sustainable-edge-vision-code

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Data & Checkpoints

- **Dataset:** This project uses the public [PlantVillage dataset](https://arxiv.org/abs/1604.03169) (38 classes). Download it and set the dataset path at the top of `01_baseline_before_optimization/notebooks/01_dataset_setup_and_split.ipynb` before running.
- **Checkpoints:** Trained model weights are not included in this repository. They are available from the author upon reasonable request.

## Reproducing the Results

Run the notebooks/scripts in the following order:

1. **Dataset setup** — `01_baseline_before_optimization/notebooks/01_dataset_setup_and_split.ipynb`
2. **Baseline training** — `01_baseline_before_optimization/notebooks/02_mobilenetv2_baseline_training.ipynb`
3. **Baseline CPU evaluation** — `01_baseline_before_optimization/scripts/evaluate_baseline_cpu.py`, `measure_baseline_latency_energy.py`, then `create_official_baseline_final_results.py`
4. **Structured pruning search + fine-tuning (Stage 1)** — `02_after_pruning_quantization/notebooks/04_pruning_finetuning_colab.ipynb`
5. **Safe pointwise incremental pruning + knowledge distillation (Stage 2)** — `02_after_pruning_quantization/advanced_pruning_experiments/notebooks/07_advanced_energy_aware_pruning_experiment_colab.ipynb`
6. **Final CPU measurement (latency / energy / CO₂)** — `02_after_pruning_quantization/advanced_pruning_experiments/scripts/final_measure_advanced_pruned_cpu_bgd_offline.py`
7. **(Optional) INT8 quantization experiments** — `02_after_pruning_quantization/notebooks/05_int8_quantization_colab.ipynb`
8. **Final baseline-vs-final comparison** — `03_final_comparison/scripts/final_compare_baseline_official_vs_advanced.py`

Each script writes its outputs (CSVs, JSON metrics, and figures) alongside the corresponding stage's folder.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{islam2026sustainableedgevision,
  title     = {Sustainable Edge Vision: An Energy-efficient Deep Learning for Crop Disease Detection},
  author    = {Islam, Md Ashraful},
  booktitle = {Proceedings of IEEE i-COSTE 2026},
  year      = {2026}
}
```

*(BibTeX entry will be updated with final page/DOI information once available in the conference proceedings.)*

## License

No license has been specified yet. Until a license file is added, please contact the author before reusing this code beyond personal inspection.

## Contact

Md Ashraful Islam — for questions, checkpoint requests, or collaboration inquiries, please open a GitHub issue or reach out directly.