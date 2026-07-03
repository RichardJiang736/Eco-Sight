#!/usr/bin/env python3
"""Power/energy measurement for Eco-Sight.

Three strategies (auto-detected by availability):
  A. CodeCarbon -- estimates energy (kWh) and emissions (kgCO2)
  B. FLOPs proxy -- model complexity via thop (always available)
  C. powermetrics -- macOS-native hardware power sampling (requires sudo)
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eco_sight.config import MODEL_DIR
from eco_sight.models import load_all_models, load_model
from eco_sight.detector import count_relevant_detections
from eco_sight.switching import select_model


def _check_strategy_a():
    try:
        import codecarbon  # noqa: F401
        return True
    except ImportError:
        return False


def _check_strategy_c():
    if platform.system() != "Darwin":
        return False
    try:
        result = subprocess.run(
            ["sudo", "-n", "powermetrics", "--samplers", "cpu_power", "-n", "1", "-i", "1000"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def _run_with_codecarbon(models, video_path, variant, fps, switching, output_dir, iteration):
    from codecarbon import EmissionsTracker

    tracker = EmissionsTracker(
        project_name=f"eco-sight-{variant}",
        output_dir=str(output_dir),
        output_file=f"codecarbon_{variant}_{iteration}.csv",
        log_level="critical",
    )

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker.start()
    t0 = time.perf_counter()

    frame_counter = 0
    switch_count = 0
    current_variant = variant
    current_fps = fps
    current_model = models[current_variant]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_counter % current_fps == 0:
            results = current_model(frame)
            if switching:
                count = count_relevant_detections(results)
                new_variant, new_fps = select_model(count)
                if new_variant != current_variant:
                    current_variant = new_variant
                    current_fps = new_fps
                    current_model = models[current_variant]
                    switch_count += 1

        frame_counter += 1

    elapsed = time.perf_counter() - t0
    emissions = tracker.stop()
    cap.release()

    energy_kwh = getattr(tracker, '_total_energy', None)
    energy_val = energy_kwh.kWh if energy_kwh else None

    return {
        "config": variant,
        "mode": "adaptive" if switching else "fixed",
        "total_frames": frame_counter,
        "elapsed_s": round(elapsed, 2),
        "effective_fps": round(frame_counter / elapsed, 2),
        "emissions_kg": round(emissions, 8) if emissions else None,
        "energy_kwh": round(energy_val, 8) if energy_val else None,
        "switch_count": switch_count,
    }


def _run_without_codecarbon(models, video_path, variant, fps, switching):
    cap = cv2.VideoCapture(str(video_path))

    t0 = time.perf_counter()
    frame_counter = 0
    switch_count = 0
    current_variant = variant
    current_fps = fps
    current_model = models[current_variant]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_counter % current_fps == 0:
            results = current_model(frame)
            if switching:
                count = count_relevant_detections(results)
                new_variant, new_fps = select_model(count)
                if new_variant != current_variant:
                    current_variant = new_variant
                    current_fps = new_fps
                    current_model = models[current_variant]
                    switch_count += 1

        frame_counter += 1

    elapsed = time.perf_counter() - t0
    cap.release()

    return {
        "config": variant,
        "mode": "adaptive" if switching else "fixed",
        "total_frames": frame_counter,
        "elapsed_s": round(elapsed, 2),
        "effective_fps": round(frame_counter / elapsed, 2),
        "emissions_kg": None,
        "energy_kwh": None,
        "switch_count": switch_count,
    }


def _measure_flops(models):
    """Strategy B: compute FLOPs and parameter counts for each model."""
    results = []
    try:
        from thop import profile
        import torch
    except ImportError:
        print("  [Strategy B] thop not installed. Skipping FLOPs measurement.")
        return results

    for variant, model in models.items():
        try:
            torch_model = model.model
            torch_model.eval()
            dummy = torch.randn(1, 3, 640, 640)
            flops, params = profile(torch_model, inputs=(dummy,), verbose=False)
            results.append({
                "variant": variant,
                "flops_g": round(flops / 1e9, 4),
                "params_m": round(params / 1e6, 2),
            })
            print(f"  {variant}: {flops / 1e9:.2f} GFLOPs, {params / 1e6:.2f}M params")
        except Exception as e:
            print(f"  {variant}: FLOPs measurement failed: {e}")

    return results


def _measure_powermetrics(duration_s: int, output_dir: Path, label: str):
    """Strategy C: sample macOS powermetrics during inference."""
    if platform.system() != "Darwin":
        return None

    sample_file = output_dir / f"powermetrics_{label}.txt"

    try:
        result = subprocess.run(
            ["sudo", "-n", "powermetrics",
             "--samplers", "cpu_power,gpu_power",
             "-n", str(max(duration_s, 5)),
             "-i", "1000",
             "-o", str(sample_file)],
            capture_output=True, text=True, timeout=duration_s + 30,
        )
        if result.returncode != 0:
            print(f"  powermetrics failed (exit {result.returncode}): {result.stderr[:200]}")
            return None

        text = sample_file.read_text() if sample_file.exists() else ""
        cpu_powers = [int(m) for m in re.findall(r"CPU Power: (\d+) mW", text)]
        gpu_powers = [int(m) for m in re.findall(r"GPU Power: (\d+) mW", text)]

        return {
            "label": label,
            "cpu_power_mw_mean": round(np.mean(cpu_powers), 1) if cpu_powers else None,
            "cpu_power_mw_max": round(np.max(cpu_powers), 1) if cpu_powers else None,
            "gpu_power_mw_mean": round(np.mean(gpu_powers), 1) if gpu_powers else None,
            "gpu_power_mw_max": round(np.max(gpu_powers), 1) if gpu_powers else None,
            "raw_file": str(sample_file),
        }
    except subprocess.TimeoutExpired:
        print("  powermetrics timed out")
        return None
    except Exception as e:
        print(f"  powermetrics error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Eco-Sight power/energy benchmark")
    parser.add_argument("--video", required=True, help="Path to test video")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations")
    parser.add_argument("--no-codecarbon", action="store_true", help="Skip CodeCarbon")
    parser.add_argument("--no-flops", action="store_true", help="Skip FLOPs measurement")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Eco-Sight Power / Energy Benchmark")
    print("=" * 60)
    print(f"  Video: {video_path}")
    print(f"  Output: {output_dir}")
    print(f"  Platform: {platform.system()} ({platform.machine()})")
    print()

    # Load models
    print("Loading models...")
    try:
        models = load_all_models()
        print(f"  Loaded: {', '.join(models.keys())}")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print("  Models are required for benchmarking. See README for instructions.")
        sys.exit(1)

    # Strategy A: CodeCarbon
    has_codecarbon = _check_strategy_a() and not args.no_codecarbon
    if has_codecarbon:
        print("\n[Strategy A] CodeCarbon energy estimation")
        print("  Requires sudo powermetrics access on macOS.")
        print("  If prompted, enter your password or Ctrl+C to skip.\n")

        all_runs = []
        configs = [
            ("nano", 5, False, "nano-only (5 FPS)"),
            ("small", 5, False, "small-only (5 FPS)"),
            ("medium", 30, False, "medium-only (30 FPS)"),
            ("small", 5, True, "adaptive switching"),
        ]

        for variant, fps, switching, label in configs:
            print(f"  Running: {label} ...")
            for i in range(args.iterations):
                try:
                    result = _run_with_codecarbon(
                        models, video_path, variant, fps, switching, output_dir, i
                    )
                    result["label"] = label
                    all_runs.append(result)
                    print(f"    Iter {i+1}: {result['elapsed_s']}s, "
                          f"energy={result['energy_kwh']} kWh, "
                          f"emissions={result['emissions_kg']} kgCO2, "
                          f"switches={result.get('switch_count', 0)}")
                except KeyboardInterrupt:
                    print("\n  Interrupted. Skipping remaining CodeCarbon runs.")
                    break
                except Exception as e:
                    print(f"    Failed: {e}")

        if all_runs:
            df = pd.DataFrame(all_runs)
            df.to_csv(output_dir / "power_codecarbon.csv", index=False)
            print(f"\n  Results saved to: {output_dir / 'power_codecarbon.csv'}")
    else:
        print("\n[Strategy A] CodeCarbon not available.")
        print("  Install with: pip install codecarbon")
        print("  On macOS, also grant sudo access to powermetrics.")
        print("  Running basic timing measurements instead...\n")

        all_runs = []
        configs = [
            ("nano", 5, False, "nano-only (5 FPS)"),
            ("small", 5, False, "small-only (5 FPS)"),
            ("medium", 30, False, "medium-only (30 FPS)"),
            ("small", 5, True, "adaptive switching"),
        ]
        for variant, fps, switching, label in configs:
            print(f"  Running: {label} ...")
            for i in range(args.iterations):
                result = _run_without_codecarbon(
                    models, video_path, variant, fps, switching
                )
                result["label"] = label
                all_runs.append(result)
                print(f"    Iter {i+1}: {result['elapsed_s']}s, "
                      f"effective FPS={result['effective_fps']}, "
                      f"switches={result.get('switch_count', 0)}")

        if all_runs:
            df = pd.DataFrame(all_runs)
            df.to_csv(output_dir / "power_timing.csv", index=False)
            print(f"\n  Results saved to: {output_dir / 'power_timing.csv'}")

    # Strategy B: FLOPs proxy
    if not args.no_flops:
        print("\n[Strategy B] FLOPs / Parameter Count")
        flops_results = _measure_flops(models)
        if flops_results:
            df = pd.DataFrame(flops_results)
            df.to_csv(output_dir / "power_flops.csv", index=False)

            # Normalize to nano as baseline
            base_flops = flops_results[0]["flops_g"]
            print("\n  Relative energy estimates (nano = 1.0x baseline):")
            for r in flops_results:
                ratio = r["flops_g"] / base_flops
                print(f"    {r['variant']}: {ratio:.2f}x FLOPs vs nano")

    # Strategy C: powermetrics (macOS only)
    if platform.system() == "Darwin":
        print("\n[Strategy C] macOS powermetrics")
        has_sudo = _check_strategy_c()
        if has_sudo:
            print("  sudo powermetrics access confirmed.")
            print("  For accurate results, avoid moving the mouse during sampling.")
            print("  Sampling during a fixed-model run for 30s ...")
            result = _measure_powermetrics(30, output_dir, "inference_sample")
            if result:
                print(f"  CPU power: mean={result['cpu_power_mw_mean']} mW, "
                      f"max={result['cpu_power_mw_max']} mW")
                if result['gpu_power_mw_mean']:
                    print(f"  GPU power: mean={result['gpu_power_mw_mean']} mW, "
                          f"max={result['gpu_power_mw_max']} mW")
                with open(output_dir / "power_powermetrics.json", "w") as f:
                    json.dump(result, f, indent=2, default=str)
        else:
            print("  powermetrics requires sudo access. To enable:")
            print("    sudo visudo")
            print("    # Add: <youruser> ALL=(ALL) NOPASSWD: /usr/bin/powermetrics")
            print("  Then re-run this benchmark.")

    print(f"\n{'=' * 60}")
    print("  Power benchmark complete.")
    print(f"  Results in: {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
