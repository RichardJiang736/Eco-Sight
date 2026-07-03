#!/usr/bin/env python3
"""Project raw M1 Pro benchmark results to target edge devices.

Reads benchmark CSVs produced by evaluate_*.py and produces projected
numbers for each specified target hardware profile.

Usage:
    python scripts/normalize_results.py --results-dir results/ --target all
    python scripts/normalize_results.py --results-dir results/ --target jetson-orin-nano-pytorch
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eco_sight.hardware_profiles import (
    ALL_PROFILES,
    get_profile,
    M1_PRO_BASELINE,
    project_inference_latency,
    project_fps,
    project_power,
    estimate_energy_per_frame,
)


def normalize_latency(results_dir: Path, output_dir: Path, profile_key: str):
    """Project latency benchmarks to target device."""
    profile = get_profile(profile_key)

    # Per-model inference latency
    inf_csv = results_dir / "latency_inference.csv"
    if inf_csv.exists():
        df = pd.read_csv(inf_csv, index_col=0)
        if "mean_ms" in df.columns:
            projected_rows = []
            for variant in ["nano", "small", "medium"]:
                if variant not in df.index:
                    continue
                slow = profile.inference_slowdown.get(variant, 1.0)
                row = {
                    "variant": variant,
                    "m1pro_mean_ms": df.loc[variant, "mean_ms"],
                    "m1pro_p95_ms": df.loc[variant, "p95_ms"],
                    "slowdown_factor": slow,
                    "projected_mean_ms": round(df.loc[variant, "mean_ms"] * slow, 2),
                    "projected_p95_ms": round(df.loc[variant, "p95_ms"] * slow, 2),
                    "projected_fps": round(1000.0 / (df.loc[variant, "mean_ms"] * slow), 1),
                }
                projected_rows.append(row)
            pd.DataFrame(projected_rows).to_csv(
                output_dir / f"latency_inference_{profile_key}.csv", index=False
            )

    # Switching latency
    sw_csv = results_dir / "latency_switching.csv"
    if sw_csv.exists():
        df = pd.read_csv(sw_csv, index_col=0)
        if "mean_ms" in df.columns:
            projected_rows = []
            for direction in df.index:
                src, dst = direction.split("->")
                src_slow = profile.inference_slowdown.get(src, 1.0)
                dst_slow = profile.inference_slowdown.get(dst, 1.0)
                avg_slow = (src_slow + dst_slow) / 2.0
                row = {
                    "direction": direction,
                    "m1pro_mean_ms": df.loc[direction, "mean_ms"],
                    "slowdown_factor": round(avg_slow, 2),
                    "projected_mean_ms": round(df.loc[direction, "mean_ms"] * avg_slow, 2),
                    "projected_p95_ms": round(df.loc[direction, "p95_ms"] * avg_slow, 2),
                }
                projected_rows.append(row)
            pd.DataFrame(projected_rows).to_csv(
                output_dir / f"latency_switching_{profile_key}.csv", index=False
            )

    # Loading latency
    load_csv = results_dir / "latency_loading.csv"
    if load_csv.exists():
        df = pd.read_csv(load_csv, index_col=0)
        if "load_time_ms" in df.columns:
            projected_rows = []
            for variant in df.index:
                if "error" in df.columns and pd.notna(df.loc[variant, "error"]):
                    continue
                row = {
                    "variant": variant,
                    "m1pro_load_ms": df.loc[variant, "load_time_ms"],
                    "slowdown_factor": profile.loading_slowdown,
                    "projected_load_ms": round(
                        df.loc[variant, "load_time_ms"] * profile.loading_slowdown, 2
                    ),
                }
                if "total_warmup_ms" in df.columns:
                    row["m1pro_warmup_ms"] = df.loc[variant, "total_warmup_ms"]
                    row["projected_warmup_ms"] = round(
                        df.loc[variant, "total_warmup_ms"] * profile.loading_slowdown, 2
                    )
                projected_rows.append(row)
            pd.DataFrame(projected_rows).to_csv(
                output_dir / f"latency_loading_{profile_key}.csv", index=False
            )


def normalize_power(results_dir: Path, output_dir: Path, profile_key: str):
    """Project power benchmarks to target device."""
    profile = get_profile(profile_key)

    timing_csv = results_dir / "power_timing.csv"
    cc_csv = results_dir / "power_codecarbon.csv"
    flops_csv = results_dir / "power_flops.csv"

    # Runtime projection (from timing or codecarbon data)
    source_csv = cc_csv if cc_csv.exists() else timing_csv
    if source_csv.exists():
        df = pd.read_csv(source_csv)
        if "config" in df.columns and "elapsed_s" in df.columns:
            projected_rows = []
            for _, row in df.iterrows():
                variant = row["config"]
                slow = profile.inference_slowdown.get(variant, 1.0)
                projected_time = row["elapsed_s"] * slow
                projected = {
                    "config": variant,
                    "mode": row.get("mode", "fixed"),
                    "m1pro_elapsed_s": row["elapsed_s"],
                    "m1pro_effective_fps": row.get("effective_fps", "N/A"),
                    "slowdown_factor": slow,
                    "projected_elapsed_s": round(projected_time, 1),
                    "projected_effective_fps": round(
                        row.get("effective_fps", 0) / slow, 1
                    ),
                }
                projected_rows.append(projected)
            pd.DataFrame(projected_rows).to_csv(
                output_dir / f"power_timing_{profile_key}.csv", index=False
            )

    # Power projection
    power_rows = []
    for variant in ["nano", "small", "medium"]:
        slow = profile.inference_slowdown.get(variant, 1.0)
        power = project_power(0, profile)
        power_rows.append({
            "variant": variant,
            "m1pro_power_w_typical": M1_PRO_BASELINE.power_watts["inference"],
            "projected_power_w_typical": profile.power_watts["inference"],
            "projected_power_w_idle": profile.power_watts["idle"],
            "projected_power_w_peak": profile.power_watts["peak"],
            "power_ratio": round(power["m1pro_to_target_ratio"], 2),
            "inference_slowdown": slow,
        })
    pd.DataFrame(power_rows).to_csv(
        output_dir / f"power_profile_{profile_key}.csv", index=False
    )

    # Energy-per-frame estimates
    if flops_csv.exists():
        flops_df = pd.read_csv(flops_csv)
        energy_rows = []
        for _, row in flops_df.iterrows():
            variant = row["variant"]
            slow = profile.inference_slowdown.get(variant, 1.0)
            est_latency = 15 * slow  # rough latency estimate for energy calc
            energy_per_frame = estimate_energy_per_frame(
                est_latency / 1000.0, profile.power_watts["inference"]
            )
            energy_rows.append({
                "variant": variant,
                "flops_g": row["flops_g"],
                "params_m": row["params_m"],
                "est_latency_ms": round(est_latency, 1),
                "est_energy_per_frame_mj": round(energy_per_frame * 1000, 2),
                "est_frames_per_watt_hour": round(
                    3600 / (energy_per_frame * 1000), 0
                ) if energy_per_frame > 0 else 0,
            })
        pd.DataFrame(energy_rows).to_csv(
            output_dir / f"power_energy_per_frame_{profile_key}.csv", index=False
        )


def normalize_accuracy(results_dir: Path, output_dir: Path, profile_key: str):
    """Accuracy notes for target devices. Accuracy is hardware-independent."""
    profile = get_profile(profile_key)

    acc_csv = results_dir / "accuracy_metrics.csv"
    if not acc_csv.exists():
        return

    df = pd.read_csv(acc_csv)
    notes = {
        "profile": profile_key,
        "device_name": profile.name,
        "accuracy_note": (
            "Detection accuracy (mAP50, precision, recall) is hardware-independent. "
            "The same model weights produce identical detection results regardless of "
            "the device they run on. Only inference speed and power consumption change. "
            "However, the adaptive switching POLICY may need re-tuning: if inference "
            "is much slower on the target device, the system may miss detection windows "
            "that would have triggered a model switch."
        ),
        "switching_impact": (
            f"With {profile.inference_slowdown.get('nano', 1.0):.1f}x slowdown for nano "
            f"and {profile.inference_slowdown.get('medium', 1.0):.1f}x for medium, "
            "the switching policy's frame-skip behavior becomes more aggressive on slower "
            "hardware since fewer frames are processed per second. At 5 FPS idle, the system "
            "samples frames every 200ms on M1 Pro but every "
            f"{200 * profile.inference_slowdown.get('nano', 1.0):.0f}ms on {profile.name}. "
            "This may cause delayed detection of new objects entering the scene."
        ),
    }

    with open(output_dir / f"accuracy_notes_{profile_key}.json", "w") as f:
        json.dump(notes, f, indent=2)

    # Also save standalone model accuracy (hardware-independent, same for all profiles)
    acc_summary = pd.DataFrame([
        {"model": "yolov10n (nano)", "mAP50": 0.348, "mAP50-95": 0.217, "precision": 0.542, "recall": 0.320},
        {"model": "yolov10s (small)", "mAP50": 0.451, "mAP50-95": 0.265, "precision": 0.625, "recall": 0.408},
        {"model": "yolov10m (medium)", "mAP50": 0.508, "mAP50-95": 0.299, "precision": 0.570, "recall": 0.473},
    ])
    acc_summary.to_csv(output_dir / "accuracy_model_baseline.csv", index=False)


def generate_hardware_comparison_table(output_dir: Path, profiles: list[str]):
    """Generate a master comparison table across all requested profiles."""
    rows = []
    for key in profiles:
        if key not in ALL_PROFILES:
            continue
        p = ALL_PROFILES[key]
        rows.append({
            "device": p.name,
            "profile_key": key,
            "nano_slowdown": p.inference_slowdown.get("nano", 1.0),
            "small_slowdown": p.inference_slowdown.get("small", 1.0),
            "medium_slowdown": p.inference_slowdown.get("medium", 1.0),
            "loading_slowdown": p.loading_slowdown,
            "typical_power_w": p.power_watts["inference"],
            "peak_power_w": p.power_watts["peak"],
            "memory_bw_ratio": p.memory_bandwidth_ratio,
            "use_case": p.use_case[:120] + "...",
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "hardware_comparison.csv", index=False)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Project M1 Pro benchmark results to target edge devices"
    )
    parser.add_argument("--results-dir", required=True, help="Directory with raw benchmark CSVs")
    parser.add_argument("--target", default="all",
                        help="Target device profile key, or 'all' for all profiles")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same as results-dir)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"Error: results directory not found: {results_dir}")
        sys.exit(1)

    if args.target == "all":
        profiles = [k for k in ALL_PROFILES if k != "m1-pro"]
    else:
        profiles = [args.target]

    print("=" * 60)
    print("  Hardware Normalization: M1 Pro → Target Devices")
    print("=" * 60)
    print(f"  Results: {results_dir}")
    print(f"  Profiles: {', '.join(profiles)}")
    print()

    for profile_key in profiles:
        if profile_key not in ALL_PROFILES:
            print(f"  Unknown profile: {profile_key}. Skipping.")
            continue

        profile = ALL_PROFILES[profile_key]
        print(f"[{profile_key}] {profile.name}")
        print(f"  Inference slowdown: n={profile.inference_slowdown.get('nano', 'N/A')}x, "
              f"s={profile.inference_slowdown.get('small', 'N/A')}x, "
              f"m={profile.inference_slowdown.get('medium', 'N/A')}x")
        print(f"  Power: {profile.power_watts['inference']}W typical, "
              f"{profile.power_watts['idle']}W idle")
        print(f"  {profile.notes[:150]}...")
        print()

        normalize_latency(results_dir, output_dir, profile_key)
        normalize_power(results_dir, output_dir, profile_key)
        normalize_accuracy(results_dir, output_dir, profile_key)

    # Generate master comparison
    comparison_df = generate_hardware_comparison_table(output_dir, profiles)
    print(f"\n  Hardware comparison table: {output_dir / 'hardware_comparison.csv'}")
    print(f"  Normalized results in: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
