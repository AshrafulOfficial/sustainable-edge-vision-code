#!/usr/bin/env python3
"""
recompute_model_size.py
Recompute baseline vs. final model size with ONE identical protocol:
FP32 weights only, saved with torch.save(model.state_dict()).

Why: the baseline size in the submitted paper (26.38 MB) came from a training
checkpoint that also holds Adam optimizer state (exp_avg / exp_avg_sq), while the
final model file (6.90 MB) holds weights only. This script removes that mismatch.

Place in:  03_final_comparison/scripts/
Run from project root, e.g.:
    python 03_final_comparison/scripts/recompute_model_size.py \
        --baseline-ckpt 01_baseline_before_optimization/models/mobilenetv2_baseline_best.pth \
        --final-model  02_after_pruning_quantization/advanced_pruning_experiments/models/safe_pointwise_kd_from_11p_extra_2percent_full_model_NOT_FINAL.pth
"""
import argparse
import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

MIB = 1024 * 1024


def load_baseline(ckpt_path, num_classes=38):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    return model.eval()


def load_final(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        return obj.eval()
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"].eval()
    raise RuntimeError(f"Unsupported final-model format: {type(obj)}")


def measure(model, tmpdir, name):
    model = model.float().cpu()
    path = Path(tmpdir) / f"{name}_state_dict_fp32.pth"
    torch.save(model.state_dict(), path)
    params = sum(p.numel() for p in model.parameters())
    payload = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    return {
        "parameters": params,
        "state_dict_file_MiB": path.stat().st_size / MIB,
        "state_dict_payload_MiB": payload / MIB,
        "parameters_x4bytes_MiB": params * 4 / MIB,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-ckpt", required=True)
    ap.add_argument("--final-model", required=True)
    ap.add_argument("--num-classes", type=int, default=38)
    ap.add_argument("--out", default="recomputed_model_size.json")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        base = measure(load_baseline(args.baseline_ckpt, args.num_classes), tmp, "baseline")
        final = measure(load_final(args.final_model), tmp, "final")

    def red(key):
        return (base[key] - final[key]) / base[key] * 100

    result = {
        "baseline": base,
        "final": final,
        "reduction_percent": {
            "state_dict_file": red("state_dict_file_MiB"),
            "state_dict_payload": red("state_dict_payload_MiB"),
            "parameters": red("parameters"),
        },
        "protocol": "FP32, torch.save(model.state_dict()), no optimizer/scheduler state",
    }
    Path(args.out).write_text(json.dumps(result, indent=2))

    print("=" * 70)
    print("Model size, identical protocol (FP32 state_dict only)")
    print("=" * 70)
    print(f"Baseline : {base['parameters']:>10,} params | file {base['state_dict_file_MiB']:.4f} MiB")
    print(f"Final    : {final['parameters']:>10,} params | file {final['state_dict_file_MiB']:.4f} MiB")
    print(f"Size reduction (file)   : {result['reduction_percent']['state_dict_file']:.2f}%")
    print(f"Size reduction (payload): {result['reduction_percent']['state_dict_payload']:.2f}%")
    print(f"Parameter reduction     : {result['reduction_percent']['parameters']:.2f}%")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()