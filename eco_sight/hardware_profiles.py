"""Hardware profiles for projecting M1 Pro benchmark results to target edge devices.

Each profile defines per-model scaling factors relative to M1 Pro MPS (PyTorch).
The scaling factors are derived from published YOLOv8 benchmarks at 640x640.
When you run benchmarks on M1 Pro, the raw results are multiplied by these factors
to estimate performance on each target device.

Sources:
  - NVIDIA Forums: Jetson Orin Nano YOLOv8 TensorRT benchmarks (2024-2025)
  - Ultralytics GitHub: M1 Pro MPS community benchmarks (discussions #4365, #2174)
  - CSDN: YOLOv8 TensorRT deployment on Jetson Orin Nano, JetPack 6.0 (2026)
  - GitHub the0807/YOLOv8-ONNX-TensorRT: Multi-model Orin Nano benchmarks
  - Luca Berton (2025): Cross-platform edge AI benchmarks

Key finding: With TensorRT FP16, Jetson Orin Nano can be FASTER than M1 Pro MPS
for nano/small models. But running the same PyTorch stack (as Eco-Sight does),
M1 Pro MPS is 3-5x faster. Accuracy (mAP50) is hardware-independent.
"""

from dataclasses import dataclass, field


@dataclass
class HardwareProfile:
    """Scaling factors for projecting M1 Pro results to a target device."""

    name: str
    description: str
    use_case: str

    # Per-model inference slowdown multipliers (vs M1 Pro MPS PyTorch).
    # inference_on_target = inference_on_m1pro * slowdown[variant]
    inference_slowdown: dict[str, float]

    # Model loading time multiplier (accounts for storage speed differences)
    loading_slowdown: float

    # Power envelope in watts {idle, inference, peak}
    power_watts: dict[str, float]

    # Memory bandwidth ratio vs M1 Pro (~200 GB/s)
    memory_bandwidth_ratio: float

    # Additional notes about the projection
    notes: str = ""


# =============================================================================
# Target Device Profiles
# =============================================================================

JETSON_ORIN_NANO_PYTORCH = HardwareProfile(
    name="Jetson Orin Nano (PyTorch)",
    description="NVIDIA Jetson Orin Nano 8GB running same Ultralytics PyTorch stack as M1 Pro.",
    use_case=(
        "Most realistic comparison: same software stack, different hardware. "
        "Represents prototyping on a laptop and deploying to the target edge device "
        "without TensorRT optimization."
    ),
    inference_slowdown={
        # M1 Pro MPS ~15ms → Jetson PyTorch ~55ms for YOLOv8n = 3.7x
        # M1 Pro MPS ~22ms → Jetson PyTorch ~80ms for YOLOv8s = 3.6x
        # M1 Pro MPS ~35ms → Jetson PyTorch ~150ms for YOLOv8m = 4.3x
        "nano": 3.7,
        "small": 3.6,
        "medium": 4.3,
    },
    loading_slowdown=1.5,  # eMMC vs M1 Pro SSD
    power_watts={"idle": 5.0, "inference": 10.0, "peak": 15.0},
    memory_bandwidth_ratio=0.34,  # 68 GB/s vs 200 GB/s
    notes=(
        "Based on PyTorch FP32 inference on Jetson Orin Nano vs M1 Pro MPS. "
        "The Orin Nano's 1024-core Ampere GPU is underutilized in PyTorch FP32 mode "
        "without TensorRT. Real-world Orin deployments should use TensorRT for 3-5x "
        "better performance (see jetson-orin-nano-tensorrt profile)."
    ),
)

JETSON_ORIN_NANO_TENSORRT = HardwareProfile(
    name="Jetson Orin Nano (TensorRT)",
    description="NVIDIA Jetson Orin Nano 8GB with TensorRT FP16 optimization.",
    use_case=(
        "Optimized edge deployment: models exported to TensorRT FP16. "
        "This is what a production autonomous driving system would use on Jetson hardware. "
        "The Orin Nano can match or beat M1 Pro MPS for small models thanks to "
        "dedicated INT8/FP16 tensor cores."
    ),
    inference_slowdown={
        # TensorRT FP16: nano ~7.3ms → 1.0x vs M1 Pro MPS (~15ms) = 0.5x FASTER
        # TensorRT FP16: small ~11.4ms vs M1 Pro MPS (~22ms) = 0.5x FASTER
        # TensorRT FP16: medium ~23.6ms vs M1 Pro MPS (~35ms) = 0.7x (slightly faster)
        "nano": 0.5,
        "small": 0.5,
        "medium": 0.7,
    },
    loading_slowdown=3.0,  # eMMC + TensorRT engine build on first load
    power_watts={"idle": 5.0, "inference": 10.0, "peak": 15.0},
    memory_bandwidth_ratio=0.34,
    notes=(
        "Based on published TensorRT FP16 benchmarks. Jetson Orin Nano achieves "
        "7.3ms for YOLOv8n, 11.4ms for YOLOv8s, 23.6ms for YOLOv8m at 640x640. "
        "These speeds require model export to TensorRT engine format (separate step). "
        "Power envelope is 7-15W configurable via nvpmodel. "
        "Sources: NVIDIA Forums, GitHub the0807/YOLOv8-ONNX-TensorRT, CSDN JetPack 6.0."
    ),
)

JETSON_XAVIER_NX = HardwareProfile(
    name="Jetson Xavier NX",
    description="NVIDIA Jetson Xavier NX 8GB with TensorRT FP16.",
    use_case=(
        "Previous-gen automotive-grade edge AI module. Still common in autonomous "
        "driving prototypes and robotics. 21 TOPS vs Orin Nano's 40 TOPS."
    ),
    inference_slowdown={
        # Xavier NX is roughly 0.6-0.7x the speed of Orin Nano
        # vs M1 Pro MPS: nano ~0.8x, small ~0.9x, medium ~1.1x
        "nano": 0.8,
        "small": 0.9,
        "medium": 1.1,
    },
    loading_slowdown=2.5,
    power_watts={"idle": 5.0, "inference": 12.0, "peak": 20.0},
    memory_bandwidth_ratio=0.26,  # 51.2 GB/s vs 200 GB/s
    notes=(
        "Xavier NX has 384 CUDA cores + 48 Tensor cores (Volta) vs Orin Nano's "
        "1024 CUDA cores + 32 Tensor cores (Ampere). Despite fewer cores, Volta "
        "tensor cores are competitive for INT8 inference. "
        "21 TOPS INT8 vs Orin Nano's 40 TOPS."
    ),
)

RASPBERRY_PI_5 = HardwareProfile(
    name="Raspberry Pi 5",
    description="Raspberry Pi 5 (8GB) with CPU-only ONNX Runtime inference.",
    use_case=(
        "Lowest-cost edge deployment. No GPU acceleration for neural networks. "
        "Represents the bottom ~20% of possible deployment targets. "
        "YOLO inference is CPU-bound on the quad-core Cortex-A76."
    ),
    inference_slowdown={
        # RPi5 CPU ONNX: nano ~300ms vs M1 Pro MPS ~15ms = 20x
        # RPi5 CPU ONNX: small ~500ms vs M1 Pro MPS ~22ms = 23x
        # RPi5 CPU ONNX: medium ~900ms vs M1 Pro MPS ~35ms = 26x
        "nano": 20.0,
        "small": 23.0,
        "medium": 26.0,
    },
    loading_slowdown=4.0,  # SD card vs SSD
    power_watts={"idle": 3.0, "inference": 8.0, "peak": 12.0},
    memory_bandwidth_ratio=0.08,  # ~16 GB/s vs 200 GB/s (LPDDR4X)
    notes=(
        "Raspberry Pi 5 has no dedicated NPU/GPU for neural network acceleration. "
        "YOLO inference runs on the CPU via ONNX Runtime or NCNN. "
        "With a Hailo-8L accelerator (13 TOPS), performance improves 5-10x "
        "but requires model compilation to Hailo's format. "
        "At these speeds, real-time adaptive switching is not practical — "
        "inference time dominates over switching overhead."
    ),
)

INTEL_NUC_I5 = HardwareProfile(
    name="Intel NUC (i5 U-series)",
    description="Intel NUC 12 with Core i5-1235U, OpenVINO backend.",
    use_case=(
        "Common x86 edge compute for industrial/automotive prototyping. "
        "Integrated Iris Xe GPU with OpenVINO optimization."
    ),
    inference_slowdown={
        # OpenVINO on i5-1235U: slightly slower than M1 Pro MPS
        # i5 U-series ~2.0x for nano, ~2.2x for small, ~2.5x for medium
        "nano": 2.0,
        "small": 2.2,
        "medium": 2.5,
    },
    loading_slowdown=1.2,  # NVMe SSD, similar to M1 Pro
    power_watts={"idle": 8.0, "inference": 20.0, "peak": 28.0},
    memory_bandwidth_ratio=0.25,  # DDR4 dual-channel ~50 GB/s
    notes=(
        "Intel NUC with OpenVINO provides good x86 inference performance. "
        "The Iris Xe iGPU (96 EU) offers some acceleration but is not as efficient "
        "as Apple's MPS or NVIDIA's TensorRT. Power consumption is higher than ARM-based "
        "edge devices. Suitable for in-vehicle PCs and industrial applications."
    ),
)

# =============================================================================
# Baseline (M1 Pro) for reference
# =============================================================================

M1_PRO_BASELINE = HardwareProfile(
    name="Apple M1 Pro (baseline)",
    description="Apple M1 Pro (16-core GPU, 16-core Neural Engine) — measurement baseline.",
    use_case="Your current benchmark machine. All scaling is relative to this device.",
    inference_slowdown={"nano": 1.0, "small": 1.0, "medium": 1.0},
    loading_slowdown=1.0,
    power_watts={"idle": 2.0, "inference": 12.0, "peak": 30.0},
    memory_bandwidth_ratio=1.0,  # 200 GB/s baseline
    notes=(
        "M1 Pro with MPS (Metal Performance Shaders) backend for PyTorch. "
        "16-core GPU at ~5.2 TFLOPS FP32. 16-core Neural Engine at ~11 TOPS. "
        "200 GB/s unified memory bandwidth. YOLO inference uses MPS (GPU), "
        "not the Neural Engine (which requires CoreML conversion)."
    ),
)

# =============================================================================
# Profile registry
# =============================================================================

ALL_PROFILES: dict[str, HardwareProfile] = {
    "m1-pro": M1_PRO_BASELINE,
    "jetson-orin-nano-pytorch": JETSON_ORIN_NANO_PYTORCH,
    "jetson-orin-nano-tensorrt": JETSON_ORIN_NANO_TENSORRT,
    "jetson-xavier-nx": JETSON_XAVIER_NX,
    "raspberry-pi-5": RASPBERRY_PI_5,
    "intel-nuc-i5": INTEL_NUC_I5,
}


def get_profile(name: str) -> HardwareProfile:
    """Get a hardware profile by key name."""
    if name not in ALL_PROFILES:
        available = ", ".join(ALL_PROFILES.keys())
        raise KeyError(f"Unknown profile '{name}'. Available: {available}")
    return ALL_PROFILES[name]


def list_profiles() -> list[dict]:
    """Return a summary of all available profiles."""
    result = []
    for key, p in ALL_PROFILES.items():
        result.append({
            "key": key,
            "name": p.name,
            "description": p.description,
            "inference_slowdown": p.inference_slowdown,
            "power_typical": p.power_watts.get("inference", "N/A"),
        })
    return result


def project_inference_latency(
    measured_ms: dict[str, float], profile: HardwareProfile
) -> dict[str, float]:
    """Project measured M1 Pro latency to target device."""
    return {
        variant: measured_ms.get(variant, 0) * profile.inference_slowdown.get(variant, 1.0)
        for variant in ["nano", "small", "medium"]
    }


def project_fps(measured_fps: dict[str, float], profile: HardwareProfile) -> dict[str, float]:
    """Project measured M1 Pro FPS to target device."""
    return {
        variant: measured_fps.get(variant, 0) / profile.inference_slowdown.get(variant, 1.0)
        for variant in ["nano", "small", "medium"]
    }


def project_power(measured_watts: float, profile: HardwareProfile) -> dict[str, float]:
    """Map M1 Pro power reading to target device power envelope.

    Since actual power measurement on M1 Pro may not reflect the same workload,
    we provide projected ranges based on the target device's published power envelope
    rather than naively scaling the M1 Pro measurement.
    """
    m1p_inference = M1_PRO_BASELINE.power_watts["inference"]
    scale = profile.power_watts["inference"] / m1p_inference if m1p_inference > 0 else 1.0
    return {
        "projected_inference_w": profile.power_watts["inference"],
        "projected_idle_w": profile.power_watts["idle"],
        "projected_peak_w": profile.power_watts["peak"],
        "m1pro_to_target_ratio": round(scale, 2),
    }


def estimate_energy_per_frame(
    latency_ms: float, power_watts: float
) -> float:
    """Estimate energy per inference frame in joules."""
    return power_watts * (latency_ms / 1000.0)
