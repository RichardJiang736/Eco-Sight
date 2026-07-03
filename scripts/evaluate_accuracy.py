#!/usr/bin/env python3
"""Accuracy comparison for Eco-Sight.

Compares adaptive model switching against three single-model baselines
(nano-only, small-only, medium-only). Uses medium-only as the ground-truth
proxy for per-frame detection agreement metrics.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eco_sight.models import load_all_models
from eco_sight.detector import count_relevant_detections, extract_relevant_boxes
from eco_sight.switching import select_model


def _box_iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a[:4]
    xb1, yb1, xb2, yb2 = box_b[:4]

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def _match_detections(det_a, det_b, iou_threshold=0.5):
    """Match boxes from det_a to det_b using greedy IoU matching."""
    if not det_a or not det_b:
        return 0, len(det_a), len(det_b)

    matched = 0
    used_b = set()
    for box_a in det_a:
        best_iou = 0
        best_j = -1
        for j, box_b in enumerate(det_b):
            if j in used_b:
                continue
            iou = _box_iou(box_a, box_b)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_threshold and best_j >= 0:
            matched += 1
            used_b.add(best_j)

    return matched, len(det_a), len(det_b)


def run_accuracy_benchmark(models, video_path: Path, max_frames: int = None):
    """Run all 4 configs frame-by-frame and compare."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)

    results = {
        "frame_idx": [],
        "nano_count": [],
        "small_count": [],
        "medium_count": [],
        "adaptive_count": [],
        "adaptive_model": [],
        "adaptive_fps": [],
        "switch_event": [],
    }

    # Adaptive state
    current_variant = "small"
    current_fps = 5
    current_model = models[current_variant]

    print(f"Processing {total_frames} frames...")

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % current_fps == 0:
            # Run all three single-model configs on same frame
            nano_r = models["nano"](frame)
            small_r = models["small"](frame)
            medium_r = models["medium"](frame)

            # Run adaptive
            adaptive_r = current_model(frame)
            count = count_relevant_detections(adaptive_r)
            new_variant, new_fps = select_model(count)

            switch = new_variant != current_variant
            if switch:
                current_variant = new_variant
                current_fps = new_fps
                current_model = models[current_variant]

            results["frame_idx"].append(frame_idx)
            results["nano_count"].append(count_relevant_detections(nano_r))
            results["small_count"].append(count_relevant_detections(small_r))
            results["medium_count"].append(count_relevant_detections(medium_r))
            results["adaptive_count"].append(count)
            results["adaptive_model"].append(current_variant)
            results["adaptive_fps"].append(current_fps)
            results["switch_event"].append(1 if switch else 0)

        if (frame_idx + 1) % 100 == 0:
            print(f"  Frame {frame_idx + 1}/{total_frames}")

    cap.release()
    return pd.DataFrame(results)


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute comparison metrics against medium-only baseline."""
    medium = df["medium_count"].values
    nano = df["nano_count"].values
    small = df["small_count"].values
    adaptive = df["adaptive_count"].values

    metrics = {}

    # Detection count MAE vs medium
    metrics["nano_count_mae"] = round(float(np.mean(np.abs(nano - medium))), 3)
    metrics["small_count_mae"] = round(float(np.mean(np.abs(small - medium))), 3)
    metrics["adaptive_count_mae"] = round(float(np.mean(np.abs(adaptive - medium))), 3)

    # Detection count correlation vs medium
    def safe_corr(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return round(float(np.corrcoef(a, b)[0, 1]), 3)

    metrics["nano_count_corr"] = safe_corr(nano, medium)
    metrics["small_count_corr"] = safe_corr(small, medium)
    metrics["adaptive_count_corr"] = safe_corr(adaptive, medium)

    # Exact agreement rate
    metrics["nano_exact_agreement"] = round(float(np.mean(nano == medium)) * 100, 1)
    metrics["small_exact_agreement"] = round(float(np.mean(small == medium)) * 100, 1)
    metrics["adaptive_exact_agreement"] = round(float(np.mean(adaptive == medium)) * 100, 1)

    # Switching behavior
    total_inferred = len(df)
    total_switches = df["switch_event"].sum()
    metrics["total_inferred_frames"] = total_inferred
    metrics["total_switches"] = int(total_switches)
    metrics["switch_rate_per_100"] = round(float(total_switches / total_inferred * 100), 1)

    # Model distribution in adaptive mode
    model_dist = df["adaptive_model"].value_counts()
    for v in ["nano", "small", "medium"]:
        metrics[f"adaptive_{v}_pct"] = round(
            float(model_dist.get(v, 0) / total_inferred * 100), 1
        )

    # Policy accuracy: for each frame, was the model selected the same
    # as what the policy would pick given medium's detection count?
    correct_selections = 0
    for _, row in df.iterrows():
        expected_variant, _ = select_model(int(row["medium_count"]))
        if row["adaptive_model"] == expected_variant:
            correct_selections += 1
    metrics["policy_selection_accuracy"] = round(
        float(correct_selections / total_inferred * 100), 1
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Eco-Sight accuracy benchmark")
    parser.add_argument("--video", required=True, help="Path to test video")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations (for CLI compat)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Eco-Sight Accuracy Benchmark")
    print("=" * 60)
    print(f"  Video: {video_path}")
    print()

    print("Loading models...")
    try:
        models = load_all_models()
        print(f"  Loaded: {', '.join(models.keys())}\n")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    # Run benchmark
    df = run_accuracy_benchmark(models, video_path, args.max_frames)

    # Save per-frame data
    df.to_csv(output_dir / "accuracy_per_frame.csv", index=False)
    print(f"\n  Per-frame data saved: {output_dir / 'accuracy_per_frame.csv'}")
    print(f"  Total inferred frames: {len(df)}")

    # Compute metrics
    metrics = compute_metrics(df)

    # Standalone model accuracy from training
    metrics["nano_mAP50"] = 0.348
    metrics["small_mAP50"] = 0.451
    metrics["medium_mAP50"] = 0.508

    # Print results
    print(f"\n{'=' * 60}")
    print("  Accuracy Comparison Results")
    print(f"{'=' * 60}")
    print()
    print("  Standalone Model Accuracy (from 200-epoch training):")
    print(f"    nano  (yolov10n): mAP50 = {metrics['nano_mAP50']:.3f}")
    print(f"    small (yolov10s): mAP50 = {metrics['small_mAP50']:.3f}")
    print(f"    medium(yolov10m): mAP50 = {metrics['medium_mAP50']:.3f}")
    print()
    print("  Detection Count vs Medium Baseline (per-frame):")
    print(f"    nano     MAE: {metrics['nano_count_mae']:.3f},  "
          f"corr: {metrics['nano_count_corr']:.3f},  "
          f"exact: {metrics['nano_exact_agreement']:.1f}%")
    print(f"    small    MAE: {metrics['small_count_mae']:.3f},  "
          f"corr: {metrics['small_count_corr']:.3f},  "
          f"exact: {metrics['small_exact_agreement']:.1f}%")
    print(f"    adaptive MAE: {metrics['adaptive_count_mae']:.3f},  "
          f"corr: {metrics['adaptive_count_corr']:.3f},  "
          f"exact: {metrics['adaptive_exact_agreement']:.1f}%")
    print()
    print("  Switching Behavior:")
    print(f"    Total switches: {metrics['total_switches']}")
    print(f"    Switch rate: {metrics['switch_rate_per_100']} per 100 frames")
    print(f"    Model distribution: "
          f"nano={metrics['adaptive_nano_pct']:.1f}%, "
          f"small={metrics['adaptive_small_pct']:.1f}%, "
          f"medium={metrics['adaptive_medium_pct']:.1f}%")
    print(f"    Policy selection accuracy: {metrics['policy_selection_accuracy']:.1f}%")

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / "accuracy_metrics.csv", index=False)
    print(f"\n  Metrics saved: {output_dir / 'accuracy_metrics.csv'}")
    print(f"  Results in: {output_dir}/")


if __name__ == "__main__":
    main()
