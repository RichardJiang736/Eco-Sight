import argparse
import sys
from pathlib import Path

from .config import MODEL_DIR, RESULTS_DIR


def _cmd_run(args):
    import time
    import cv2
    from .models import load_all_models
    from .detector import count_relevant_detections
    from .switching import select_model

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}")
        sys.exit(1)

    models = load_all_models()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: cannot open video: {video_path}")
        sys.exit(1)

    frame_counter = 0
    current_variant = "small"
    current_fps = 5
    current_model = models[current_variant]

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log_file = open(args.log_file, "w") if args.log_file else None

    print(f"Running inference on: {video_path}")
    print(f"Total frames: {total_frames}")
    print("Press 'q' to quit.\n")

    t0 = time.perf_counter()
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_counter % current_fps == 0:
                results = current_model(frame)
                count = count_relevant_detections(results)
                new_variant, new_fps = select_model(count)

                if new_variant != current_variant:
                    current_variant = new_variant
                    current_fps = new_fps
                    current_model = models[current_variant]
                    msg = (
                        f"Frame {frame_counter}: {count} objects -> "
                        f"{current_model.name} ({current_variant}), {current_fps} FPS"
                    )
                    print(msg)
                    if log_file:
                        log_file.write(msg + "\n")

            if not args.no_display:
                cv2.imshow("Eco-Sight", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_counter += 1
    finally:
        elapsed = time.perf_counter() - t0
        cap.release()
        cv2.destroyAllWindows()
        if log_file:
            log_file.close()

    print(f"\nProcessed {frame_counter} frames in {elapsed:.1f}s "
          f"({frame_counter / elapsed:.1f} FPS effective)")


def _cmd_benchmark(args):
    import subprocess
    import os

    video = Path(args.video)
    if not video.exists():
        print(f"Error: video not found: {video}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"

    steps = [
        ("Power benchmark", "evaluate_power.py"),
        ("Latency benchmark", "evaluate_latency.py"),
        ("Accuracy benchmark", "evaluate_accuracy.py"),
    ]

    for label, script in steps:
        script_path = scripts_dir / script
        if not script_path.exists():
            print(f"  SKIP: {script} not found")
            continue
        print(f"\n{'='*60}\n  {label}\n{'='*60}")
        subprocess.run(
            [sys.executable, str(script_path),
             "--video", str(video),
             "--output-dir", str(output_dir),
             "--iterations", str(args.iterations)],
            check=False,
            env={**os.environ, "ECO_SIGHT_MODEL_DIR": os.environ.get("ECO_SIGHT_MODEL_DIR", str(MODEL_DIR))},
        )

    print(f"\nAll benchmarks complete. Results in: {output_dir}")


def _cmd_normalize(args):
    import subprocess

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: results directory not found: {results_dir}")
        sys.exit(1)

    script_path = Path(__file__).resolve().parent.parent / "scripts" / "normalize_results.py"
    if not script_path.exists():
        print(f"Error: normalize_results.py not found at {script_path}")
        sys.exit(1)

    subprocess.run(
        [sys.executable, str(script_path),
         "--results-dir", str(results_dir),
         "--target", args.target,
         "--output-dir", str(results_dir)],
        check=False,
    )


def _cmd_report(args):
    import subprocess

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: results directory not found: {results_dir}")
        sys.exit(1)

    script_path = Path(__file__).resolve().parent.parent / "scripts" / "generate_report.py"
    if not script_path.exists():
        print(f"Error: generate_report.py not found at {script_path}")
        sys.exit(1)

    subprocess.run(
        [sys.executable, str(script_path),
         "--results-dir", str(results_dir),
         "--output", str(args.output)],
        check=False,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="eco-sight",
        description="Eco-Sight: Adaptive model switching for energy-efficient perception",
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run inference on a video")
    run_p.add_argument("--video", required=True, help="Path to input video")
    run_p.add_argument("--no-display", action="store_true", help="Disable OpenCV display window")
    run_p.add_argument("--log-file", default=None, help="Path to write switching event log")

    bench_p = sub.add_parser("benchmark", help="Run full evaluation suite")
    bench_p.add_argument("--video", required=True, help="Path to test video")
    bench_p.add_argument("--output-dir", default=str(RESULTS_DIR), help="Output directory for results")
    bench_p.add_argument("--iterations", type=int, default=1, help="Number of benchmark iterations")

    norm_p = sub.add_parser("normalize", help="Project M1 Pro results to target edge devices")
    norm_p.add_argument("--results-dir", required=True, help="Directory containing benchmark CSVs")
    norm_p.add_argument("--target", default="all",
                        help="Target device profile (jetson-orin-nano-pytorch, jetson-orin-nano-tensorrt, "
                             "jetson-xavier-nx, raspberry-pi-5, intel-nuc-i5, or 'all')")

    report_p = sub.add_parser("report", help="Generate evaluation report from results")
    report_p.add_argument("--results-dir", required=True, help="Directory containing benchmark CSVs")
    report_p.add_argument("--output", default="report.md", help="Output report path")

    args = parser.parse_args()
    if args.command == "run":
        _cmd_run(args)
    elif args.command == "benchmark":
        _cmd_benchmark(args)
    elif args.command == "normalize":
        _cmd_normalize(args)
    elif args.command == "report":
        _cmd_report(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
