# Final Optimized Model Report

## Final Model
**Final Optimized MobileNetV2 - 11% Pruned + QAT INT8**

## Saved TorchScript Model
`/content/gdrive_mount/MyDrive/Thesis/thesis_project/02_after_pruning_quantization/models/Final_Optimized_MobileNetV2_11Percent_Pruned_QAT_INT8_TorchScript.pt`

## Optimization Pipeline
Baseline MobileNetV2 → 11% Structured Pruning → QAT Fine-Tuning → INT8 Conversion

## Validation Results

| Model | Validation Accuracy | Macro F1-score |
|---|---:|---:|
| Baseline MobileNetV2 | 0.996931 | 0.996551 |
| 11% Pruned FP32 MobileNetV2 | 0.986251 | 0.982698 |
| QAT Fake-Quant Model | 0.969433 | 0.960055 |
| Final QAT-INT8 TorchScript Model | 0.971151 | 0.960798 |

## Performance Drop from Baseline
- Accuracy drop: 2.5780 percentage points
- Macro F1 drop: 3.5754 percentage points

## Performance Drop from 11% Pruned FP32 Model
- Accuracy drop: 1.5099 percentage points
- Macro F1 drop: 2.1900 percentage points

## Final Model Size
- TorchScript model size: 2.4138 MB

## Next Step
Run final local CPU test-set evaluation using the same test images, preprocessing, batch size, and energy measurement procedure as the baseline.
