#!/usr/bin/env python3
"""Generate a comprehensive evaluation report from benchmark results.

Reads CSVs from a results directory and produces a Markdown report with
tables, summary statistics, and answers to the three core questions:
  1. How much power does the system save?
  2. How fast does it switch between models?
  3. How accurate is the system?
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _safe_read(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


def _write_section(f, title: str, content_fn):
    f.write(f"\n## {title}\n\n")
    content_fn()


def _write_table(f, df: pd.DataFrame, float_fmt: str = ".3f"):
    if df is None or df.empty:
        f.write("_No data available._\n\n")
        return
    f.write("\n")
    f.write(df.to_markdown(index=False, floatfmt=float_fmt))
    f.write("\n\n")


def _write_kv(f, items: list[tuple[str, str]]):
    for key, value in items:
        f.write(f"| **{key}** | {value} |\n")


def generate_report(results_dir: Path, output_path: Path):
    # Load all available data
    power_cc = _safe_read(results_dir / "power_codecarbon.csv")
    power_timing = _safe_read(results_dir / "power_timing.csv")
    power_flops = _safe_read(results_dir / "power_flops.csv")

    latency_inf = _safe_read(results_dir / "latency_inference.csv")
    latency_switch = _safe_read(results_dir / "latency_switching.csv")
    latency_load = _safe_read(results_dir / "latency_loading.csv")

    accuracy_frame = _safe_read(results_dir / "accuracy_per_frame.csv")
    accuracy_metrics = _safe_read(results_dir / "accuracy_metrics.csv")

    with open(output_path, "w") as f:
        f.write("# Eco-Sight Evaluation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(
            "This report evaluates the Eco-Sight adaptive model switching system "
            "across three dimensions: **power/energy savings**, **model switching latency**, "
            "and **detection accuracy**. Eco-Sight uses three YOLOv10 variants (nano, small, medium) "
            "and dynamically selects the active model based on the number of detected objects "
            "in each frame.\n\n"
        )

        # Determine what data we have
        has_power = power_cc is not None or power_timing is not None
        has_latency = latency_inf is not None
        has_accuracy = accuracy_metrics is not None

        summary_items = []
        if has_power:
            summary_items.append(
                ("Power", "Measured -- see Section 1 for energy comparison across configurations.")
            )
        else:
            summary_items.append(
                ("Power", "Not measured. Run `eco-sight benchmark` to collect data.")
            )
        if has_latency:
            summary_items.append(
                ("Latency", "Measured -- see Section 2 for switching speed and per-model timings.")
            )
        else:
            summary_items.append(
                ("Latency", "Not measured. Run `eco-sight benchmark` to collect data.")
            )
        if has_accuracy:
            summary_items.append(
                ("Accuracy", "Measured -- see Section 3 for adaptive vs single-model comparison.")
            )
        else:
            summary_items.append(
                ("Accuracy", "Not measured. Run `eco-sight benchmark` to collect data.")
            )
        _write_kv(f, summary_items)
        f.write("\n")

        # Section 1: Power
        f.write("---\n")
        f.write("## 1. Power / Energy Savings\n\n")

        if power_cc is not None:
            f.write("### CodeCarbon Measurements\n\n")
            f.write("Energy consumption and CO2 emissions estimated via CodeCarbon.\n")
            _write_table(f, power_cc)

            if "energy_kwh" in power_cc.columns and "config" in power_cc.columns:
                adapt = power_cc[power_cc["mode"] == "adaptive"]
                fixed = power_cc[power_cc["mode"] == "fixed"]
                if not adapt.empty and not fixed.empty:
                    adapt_energy = adapt["energy_kwh"].mean()
                    fixed_energy = fixed["energy_kwh"].mean()
                    if adapt_energy and fixed_energy:
                        ratio = (fixed_energy - adapt_energy) / fixed_energy * 100
                        f.write(f"**Energy comparison:** Adaptive mode uses ")
                        f.write(f"{adapt_energy:.6f} kWh vs fixed-mode average ")
                        f.write(f"{fixed_energy:.6f} kWh ")
                        if ratio > 0:
                            f.write(f"({ratio:.1f}% reduction with adaptive switching).\n\n")
                        else:
                            f.write(f"(no net savings detected).\n\n")

        elif power_timing is not None:
            f.write("### Timing Measurements\n\n")
            f.write("Basic runtime measurements (CodeCarbon not available).\n")
            _write_table(f, power_timing)

            adapt = power_timing[power_timing["mode"] == "adaptive"]
            fixed = power_timing[power_timing["mode"] == "fixed"]
            if not adapt.empty and not fixed.empty:
                adapt_time = adapt["elapsed_s"].mean()
                fixed_time = fixed["elapsed_s"].mean()
                f.write(f"**Runtime comparison:** Adaptive: {adapt_time:.1f}s ")
                f.write(f"vs fixed-mode average: {fixed_time:.1f}s.\n\n")

        if power_flops is not None:
            f.write("### Model FLOPs / Parameter Count\n\n")
            f.write("Computational complexity of each model variant (proxy for relative energy use).\n")
            _write_table(f, power_flops)

            if len(power_flops) >= 3:
                flops_n = power_flops[power_flops["variant"] == "nano"]["flops_g"].values[0]
                flops_s = power_flops[power_flops["variant"] == "small"]["flops_g"].values[0]
                flops_m = power_flops[power_flops["variant"] == "medium"]["flops_g"].values[0]
                f.write(f"**Relative FLOPs:** small = {flops_s/flops_n:.1f}x nano, ")
                f.write(f"medium = {flops_m/flops_n:.1f}x nano.\n\n")
        else:
            f.write("_Model complexity data not available. Run benchmark to collect._\n\n")

        # Section 2: Latency
        f.write("---\n")
        f.write("## 2. Model Switching Latency\n\n")

        if latency_inf is not None:
            f.write("### Per-Model Inference Latency\n\n")
            f.write("Steady-state single-frame inference time for each model variant.\n")
            inf_display = latency_inf[["mean_ms", "std_ms", "p50_ms", "p95_ms", "p99_ms"]] if "mean_ms" in latency_inf.columns else latency_inf
            f.write("\n")
            f.write(latency_inf.to_markdown(floatfmt=".2f"))
            f.write("\n\n")
        else:
            f.write("_Inference latency data not available._\n\n")

        if latency_switch is not None:
            f.write("### Model Switching Latency\n\n")
            f.write("Time from last inference with source model to first inference with destination model.\n")
            f.write("\n")
            f.write(latency_switch.to_markdown(floatfmt=".2f"))
            f.write("\n\n")

            if "mean_ms" in latency_switch.columns:
                fastest = latency_switch["mean_ms"].idxmin()
                slowest = latency_switch["mean_ms"].idxmax()
                f.write(f"**Fastest switch:** {fastest} ({latency_switch.loc[fastest, 'mean_ms']:.2f} ms)\n\n")
                f.write(f"**Slowest switch:** {slowest} ({latency_switch.loc[slowest, 'mean_ms']:.2f} ms)\n\n")
        else:
            f.write("_Switching latency data not available._\n\n")

        if latency_load is not None:
            f.write("### Model Loading Latency (Cold Start)\n\n")
            f.write("Time to load each model from disk and run first inference.\n")
            f.write("\n")
            f.write(latency_load.to_markdown(floatfmt=".2f"))
            f.write("\n\n")
        else:
            f.write("_Loading latency data not available._\n\n")

        # Section 3: Accuracy
        f.write("---\n")
        f.write("## 3. Detection Accuracy\n\n")

        f.write("### Standalone Model Accuracy (from 200-epoch training)\n\n")
        f.write("| Model | mAP50 | mAP50-95 | Precision | Recall |\n")
        f.write("|-------|-------|----------|-----------|--------|\n")
        f.write("| yolov10n (nano) | 0.348 | 0.217 | 0.542 | 0.320 |\n")
        f.write("| yolov10s (small) | 0.451 | 0.265 | 0.625 | 0.408 |\n")
        f.write("| yolov10m (medium) | 0.508 | 0.299 | 0.570 | 0.473 |\n\n")
        f.write("These are validation-set mAP values from the 200-epoch fine-tuning runs on the nuScenes-derived dataset.\n\n")

        if accuracy_metrics is not None:
            f.write("### Adaptive vs Single-Model Comparison\n\n")
            f.write("Using medium-only (best standalone model) as the ground-truth proxy:\n\n")

            metrics = accuracy_metrics.iloc[0].to_dict()
            f.write("| Metric | nano | small | adaptive |\n")
            f.write("|--------|------|-------|----------|\n")
            f.write(f"| Count MAE vs medium | {metrics.get('nano_count_mae', 'N/A')} | "
                    f"{metrics.get('small_count_mae', 'N/A')} | "
                    f"{metrics.get('adaptive_count_mae', 'N/A')} |\n")
            f.write(f"| Count correlation vs medium | {metrics.get('nano_count_corr', 'N/A')} | "
                    f"{metrics.get('small_count_corr', 'N/A')} | "
                    f"{metrics.get('adaptive_count_corr', 'N/A')} |\n")
            f.write(f"| Exact count agreement | {metrics.get('nano_exact_agreement', 'N/A')}% | "
                    f"{metrics.get('small_exact_agreement', 'N/A')}% | "
                    f"{metrics.get('adaptive_exact_agreement', 'N/A')}% |\n")
            f.write("\n")

            f.write("### Switching Behavior\n\n")
            f.write(f"- **Total switches:** {metrics.get('total_switches', 'N/A')}\n")
            f.write(f"- **Switch rate:** {metrics.get('switch_rate_per_100', 'N/A')} per 100 inferred frames\n")
            f.write(f"- **Model distribution:** nano={metrics.get('adaptive_nano_pct', 'N/A')}%, "
                    f"small={metrics.get('adaptive_small_pct', 'N/A')}%, "
                    f"medium={metrics.get('adaptive_medium_pct', 'N/A')}%\n")
            f.write(f"- **Policy selection accuracy:** {metrics.get('policy_selection_accuracy', 'N/A')}% "
                    f"(fraction of frames where adaptive selected the model predicted by the policy given "
                    f"medium's detection count)\n\n")
        else:
            f.write("_Adaptive accuracy comparison data not available. Run benchmark to collect._\n\n")

        # Section 4: Answers to Core Questions
        f.write("---\n")
        f.write("## 4. Answers to Core Questions\n\n")

        f.write("### Q1: How much power does the system save?\n\n")
        if power_cc is not None and "energy_kwh" in power_cc.columns:
            adapt = power_cc[power_cc["mode"] == "adaptive"]
            medium_only = power_cc[(power_cc["mode"] == "fixed") & (power_cc["config"] == "medium")]
            if not adapt.empty and not medium_only.empty:
                a_e = adapt["energy_kwh"].mean()
                m_e = medium_only["energy_kwh"].mean()
                if a_e and m_e:
                    savings = (m_e - a_e) / m_e * 100
                    f.write(f"Adaptive switching saves **{savings:.1f}%** energy compared to running "
                            f"the medium model exclusively ({a_e:.6f} kWh vs {m_e:.6f} kWh). ")
                else:
                    f.write("Energy could not be estimated. ")
        else:
            f.write("Power data is not available. ")

        if power_flops is not None and len(power_flops) >= 3:
            flops = {r["variant"]: r["flops_g"] for _, r in power_flops.iterrows()}
            f.write(f"Based on FLOPs analysis, nano uses {flops['nano']:.1f} GFLOPs vs "
                    f"medium at {flops['medium']:.1f} GFLOPs ({flops['medium']/flops['nano']:.1f}x difference). ")
            f.write("When the system spends significant time in nano/low-FPS mode, "
                    "energy savings scale proportionally to the reduction in compute.\n\n")
        else:
            f.write("Run the benchmark with FLOPs measurement enabled for model-complexity estimates.\n\n")

        f.write("### Q2: How fast does the system switch between models?\n\n")
        if latency_switch is not None and "mean_ms" in latency_switch.columns:
            avg_switch = latency_switch["mean_ms"].mean()
            f.write(f"Average model switching latency is **{avg_switch:.2f} ms** across all 6 directions. ")
            f.write("Since all models are pre-loaded into memory, the switch cost is essentially "
                    "the inference time of the destination model. ")
            f.write("The switching is effectively instant -- there is no disk I/O or model reloading "
                    "during operation.\n\n")
        elif latency_inf is not None and "mean_ms" in latency_inf.columns:
            avg_inf = latency_inf["mean_ms"].mean()
            f.write(f"Based on inference latency (avg {avg_inf:.2f} ms), switching speed is bounded by "
                    f"the destination model's per-frame time. With pre-loaded models, switching is immediate.\n\n")
        else:
            f.write("Latency data is not available. Run benchmark to collect.\n\n")

        f.write("### Q3: How accurate is the system?\n\n")
        f.write("**Standalone accuracy (200-epoch training, nuScenes validation):**\n")
        f.write("- nano:  mAP50 = 0.348 (34.8%)\n")
        f.write("- small: mAP50 = 0.451 (45.1%)\n")
        f.write("- medium: mAP50 = 0.508 (50.8%)\n\n")

        if accuracy_metrics is not None:
            metrics = accuracy_metrics.iloc[0].to_dict()
            mae = metrics.get("adaptive_count_mae", "N/A")
            corr = metrics.get("adaptive_count_corr", "N/A")
            agree = metrics.get("adaptive_exact_agreement", "N/A")
            f.write(f"**Adaptive switching accuracy (vs medium ground truth):**\n")
            f.write(f"- Detection count MAE: {mae}\n")
            f.write(f"- Detection count correlation: {corr}\n")
            f.write(f"- Exact count agreement rate: {agree}%\n")
            f.write(f"- Policy selection accuracy: {metrics.get('policy_selection_accuracy', 'N/A')}%\n\n")
            f.write("These results show how closely the adaptive system tracks the most accurate single model. ")
            f.write("Note: this is a relative comparison, not an absolute accuracy metric.\n\n")
        else:
            f.write("Adaptive accuracy comparison data is not available. Run benchmark to collect.\n\n")

        # Section 5: Hardware Projections (if normalized data exists)
        hw_comparison = _safe_read(results_dir / "hardware_comparison.csv")
        latency_proj_files = list(results_dir.glob("latency_inference_jetson*.csv"))
        power_proj_files = list(results_dir.glob("power_timing_jetson*.csv"))

        if hw_comparison is not None or latency_proj_files or power_proj_files:
            f.write("---\n")
            f.write("## 5. Hardware Projections: From M1 Pro to Target Devices\n\n")
            f.write(
                "_This section projects the M1 Pro benchmark results to realistic edge deployment "
                "targets. Inference speed and power scale with hardware; accuracy (mAP50) is "
                "hardware-independent and does not change. All projections are estimates based on "
                "published YOLO benchmarks across platforms._\n\n"
            )

            if hw_comparison is not None:
                f.write("### Target Device Profiles\n\n")
                f.write("Scaling factors relative to M1 Pro MPS baseline:\n")
                display_cols = ["device", "nano_slowdown", "small_slowdown",
                                "medium_slowdown", "typical_power_w"]
                hw_display = hw_comparison[display_cols].copy()
                hw_display.columns = ["Device", "Nano Slowdown", "Small Slowdown",
                                       "Medium Slowdown", "Typical Power (W)"]
                _write_table(f, hw_display, float_fmt=".1f")

            # Per-device latency projections
            for proj_file in sorted(latency_proj_files):
                df = _safe_read(proj_file)
                if df is None:
                    continue
                device_name = proj_file.stem.replace("latency_inference_", "").replace("-", " ").title()
                f.write(f"### Inference Latency: {device_name}\n\n")
                _write_table(f, df)
                if "projected_fps" in df.columns:
                    fps_n = df[df["variant"] == "nano"]["projected_fps"].values
                    fps_s = df[df["variant"] == "small"]["projected_fps"].values
                    fps_m = df[df["variant"] == "medium"]["projected_fps"].values
                    if len(fps_n) > 0:
                        f.write(f"**Projected FPS:** nano={fps_n[0]:.1f}, "
                                f"small={fps_s[0]:.1f}, medium={fps_m[0]:.1f}. ")
                        if fps_n[0] < 30:
                            f.write("Nano model achieves real-time (30+ FPS) only if it's less than 33ms. ")
                        f.write("\n\n")

            # Per-device power projections
            for proj_file in sorted(power_proj_files):
                df = _safe_read(proj_file)
                if df is None:
                    continue
                device_name = proj_file.stem.replace("power_timing_", "").replace("-", " ").title()
                f.write(f"### Runtime Projection: {device_name}\n\n")
                _write_table(f, df)

                if "projected_elapsed_s" in df.columns and "mode" in df.columns:
                    adapt = df[df["mode"] == "adaptive"]
                    fixed = df[df["mode"] == "fixed"]
                    if not adapt.empty and not fixed.empty:
                        adapt_time = adapt["projected_elapsed_s"].mean()
                        fixed_times = fixed["projected_elapsed_s"].mean()
                        f.write(f"**On this device:** Adaptive mode projected at "
                                f"{adapt_time:.1f}s vs fixed-mode avg {fixed_times:.1f}s.\n\n")

            # Accuracy note
            f.write("### Accuracy on Target Devices\n\n")
            f.write(
                "**Accuracy is hardware-independent.** The same model weights produce identical "
                "detection results on any device. The mAP50 values (nano=0.348, small=0.451, "
                "medium=0.508) are constant across hardware.\n\n"
                "However, the **adaptive switching policy should be re-tuned** for the target "
                "device's inference speed. On slower hardware:\n"
                "- Frame-skipping at 5 FPS means larger time gaps between inferences, which may "
                "delay detection of new objects\n"
                "- The 30 FPS \"high energy\" mode may be unachievable on slow devices; the policy "
                "should cap FPS at the device's maximum\n"
                "- Model loading from slower storage (eMMC/SD card) increases startup latency "
                "but doesn't affect runtime switching\n\n"
            )

            # Energy savings projection summary
            energy_files = list(results_dir.glob("power_energy_per_frame_*.csv"))
            if energy_files:
                f.write("### Energy Efficiency Comparison\n\n")
                f.write("Estimated energy per inference frame across devices:\n\n")
                f.write("| Device | Nano (mJ/frame) | Small (mJ/frame) | Medium (mJ/frame) |\n")
                f.write("|--------|-----------------|------------------|-------------------|\n")

                m1p_inf = _safe_read(results_dir / "latency_inference.csv")
                m1p_latency = {}
                if m1p_inf is not None and "mean_ms" in m1p_inf.columns:
                    for v in ["nano", "small", "medium"]:
                        if v in m1p_inf.index:
                            m1p_latency[v] = m1p_inf.loc[v, "mean_ms"]

                from eco_sight.hardware_profiles import M1_PRO_BASELINE, estimate_energy_per_frame
                m1p_power = M1_PRO_BASELINE.power_watts["inference"]
                m1p_energies = {}
                for v in ["nano", "small", "medium"]:
                    if v in m1p_latency:
                        m1p_energies[v] = estimate_energy_per_frame(m1p_latency[v] / 1000, m1p_power) * 1000

                if m1p_energies:
                    f.write(f"| M1 Pro (baseline) | {m1p_energies.get('nano', 'N/A'):.1f} | "
                            f"{m1p_energies.get('small', 'N/A'):.1f} | "
                            f"{m1p_energies.get('medium', 'N/A'):.1f} |\n")

                for ef in sorted(energy_files):
                    df = _safe_read(ef)
                    if df is None or "est_energy_per_frame_mj" not in df.columns:
                        continue
                    device = ef.stem.replace("power_energy_per_frame_", "").replace("-", " ").title()
                    energies = {}
                    for _, row in df.iterrows():
                        energies[row["variant"]] = row["est_energy_per_frame_mj"]
                    f.write(f"| {device} | {energies.get('nano', 'N/A'):.1f} | "
                            f"{energies.get('small', 'N/A'):.1f} | "
                            f"{energies.get('medium', 'N/A'):.1f} |\n")
                f.write("\n")

        # Footer
        f.write("---\n")
        f.write(f"\n_Report generated by Eco-Sight evaluation suite v0.2.0_\n"
                "_Hardware projections are estimates based on published benchmarks. "
                "Validate with real hardware for production decisions._\n")

    print(f"Report written to: {output_path}")
    print(f"Size: {output_path.stat().st_size:,} bytes")


def main():
    parser = argparse.ArgumentParser(description="Generate Eco-Sight evaluation report")
    parser.add_argument("--results-dir", required=True, help="Directory with benchmark CSVs")
    parser.add_argument("--output", default="report.md", help="Output report path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)

    if not results_dir.exists():
        print(f"Error: results directory not found: {results_dir}")
        print("Run 'eco-sight benchmark' first to generate benchmark data.")
        sys.exit(1)

    csv_count = len(list(results_dir.glob("*.csv")))
    if csv_count == 0:
        print(f"Warning: no CSV files found in {results_dir}")
        print("The report will contain only static training data.")

    generate_report(results_dir, output_path)


if __name__ == "__main__":
    main()
