#!/usr/bin/env python3
"""Model switching latency benchmark for Eco-Sight.

Measures:
  1. Per-model steady-state inference latency (100+ frames each)
  2. Model switching latency (all 6 directions, 50 iterations each)
  3. Model loading latency (cold-start from disk)
  4. First-inference warmup time
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eco_sight.models import load_all_models, load_model


def _read_single_frame(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame from {video_path}")
    return frame


def measure_inference_latency(models, video_path: Path, num_frames: int = 100):
    """Measure steady-state per-frame inference time for each model."""
    frame = _read_single_frame(video_path)
    results = {}

    for variant, model in models.items():
        latencies = []
        for _ in range(5):
            _ = model(frame)
        for _ in range(num_frames):
            t0 = time.perf_counter()
            _ = model(frame)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

        arr = np.array(latencies)
        results[variant] = {
            "mean_ms": round(np.mean(arr), 2),
            "std_ms": round(np.std(arr), 2),
            "min_ms": round(np.min(arr), 2),
            "max_ms": round(np.max(arr), 2),
            "p50_ms": round(np.percentile(arr, 50), 2),
            "p95_ms": round(np.percentile(arr, 95), 2),
            "p99_ms": round(np.percentile(arr, 99), 2),
            "num_samples": num_frames,
        }
        print(f"  {variant}: mean={results[variant]['mean_ms']:.2f}ms, "
              f"p95={results[variant]['p95_ms']:.2f}ms, "
              f"p99={results[variant]['p99_ms']:.2f}ms")

    return results


def measure_switching_latency(models, video_path: Path, num_iterations: int = 50):
    """Measure model switch cost: time from last src-model inference to first dst-model inference."""
    frame = _read_single_frame(video_path)
    variants = list(models.keys())
    results = {}

    for src in variants:
        for dst in variants:
            if src == dst:
                continue
            key = f"{src}->{dst}"
            latencies = []
            for _ in range(num_iterations):
                _ = models[src](frame)
                _ = models[dst](frame)

                _ = models[src](frame)
                t0 = time.perf_counter()
                _ = models[dst](frame)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)

            arr = np.array(latencies)
            results[key] = {
                "mean_ms": round(np.mean(arr), 2),
                "std_ms": round(np.std(arr), 2),
                "min_ms": round(np.min(arr), 2),
                "max_ms": round(np.max(arr), 2),
                "p50_ms": round(np.percentile(arr, 50), 2),
                "p95_ms": round(np.percentile(arr, 95), 2),
                "p99_ms": round(np.percentile(arr, 99), 2),
                "num_iterations": num_iterations,
            }
            print(f"  {key}: mean={results[key]['mean_ms']:.2f}ms, "
                  f"p95={results[key]['p95_ms']:.2f}ms")

    return results


def measure_loading_latency():
    """Measure time to load each model from disk (cold start)."""
    from eco_sight.config import MODEL_VARIANTS

    results = {}
    for variant in MODEL_VARIANTS:
        t0 = time.perf_counter()
        try:
            model = load_model(variant)
            t1 = time.perf_counter()
            results[variant] = {
                "load_time_ms": round((t1 - t0) * 1000, 2),
            }
            frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            t2 = time.perf_counter()
            _ = model(frame)
            t3 = time.perf_counter()
            results[variant]["first_inference_ms"] = round((t3 - t2) * 1000, 2)
            results[variant]["total_warmup_ms"] = round((t3 - t0) * 1000, 2)
            print(f"  {variant}: load={results[variant]['load_time_ms']:.0f}ms, "
                  f"first_inference={results[variant]['first_inference_ms']:.1f}ms")
        except FileNotFoundError as e:
            print(f"  {variant}: SKIP - {e}")
            results[variant] = {"error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(description="Eco-Sight switching latency benchmark")
    parser.add_argument("--video", required=True, help="Path to test video")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--inference-frames", type=int, default=100, help="Frames per model for steady-state measurement")
    parser.add_argument("--switch-iterations", type=int, default=50, help="Iterations per switch direction")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Eco-Sight Switching Latency Benchmark")
    print("=" * 60)
    print(f"  Video: {video_path}")
    print()

    # Load models
    print("Loading models...")
    try:
        models = load_all_models()
        print(f"  Loaded: {', '.join(models.keys())}\n")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    # 1. Steady-state inference latency
    print("[1/3] Per-model inference latency")
    print("-" * 40)
    inference_results = measure_inference_latency(models, video_path, args.inference_frames)
    df_inf = pd.DataFrame(inference_results).T
    df_inf.index.name = "variant"
    df_inf.to_csv(output_dir / "latency_inference.csv")
    print(f"  Saved: {output_dir / 'latency_inference.csv'}\n")

    # 2. Switching latency
    print("[2/3] Model switching latency")
    print("-" * 40)
    switch_results = measure_switching_latency(models, video_path, args.switch_iterations)
    df_switch = pd.DataFrame(switch_results).T
    df_switch.index.name = "direction"
    df_switch.to_csv(output_dir / "latency_switching.csv")
    print(f"  Saved: {output_dir / 'latency_switching.csv'}\n")

    # 3. Loading latency
    print("[3/3] Model loading latency (cold start)")
    print("-" * 40)
    loading_results = measure_loading_latency()
    df_load = pd.DataFrame(loading_results).T
    df_load.index.name = "variant"
    df_load.to_csv(output_dir / "latency_loading.csv")
    print(f"  Saved: {output_dir / 'latency_loading.csv'}\n")

    # Summary
    print("=" * 60)
    print("  Latency Summary")
    print("=" * 60)
    for v in ["nano", "small", "medium"]:
        if v in inference_results:
            print(f"  {v}: {inference_results[v]['mean_ms']}ms avg inference "
                  f"(p95: {inference_results[v]['p95_ms']}ms)")
    print(f"\n  Fastest switch: {min(switch_results, key=lambda k: switch_results[k]['mean_ms'])} "
          f"({switch_results[min(switch_results, key=lambda k: switch_results[k]['mean_ms'])]['mean_ms']}ms)")
    print(f"  Slowest switch: {max(switch_results, key=lambda k: switch_results[k]['mean_ms'])} "
          f"({switch_results[max(switch_results, key=lambda k: switch_results[k]['mean_ms'])]['mean_ms']}ms)")
    print(f"\n  Results in: {output_dir}/")


if __name__ == "__main__":
    main()
