#!/usr/bin/env python3
"""Run the per-GPU CUDA benchmark and build TXT, CSV, and JSON reports."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import json
import pathlib
import shlex
import subprocess
import sys
from typing import Any


DEFAULT_PRECISIONS = "int4,int8,fp8_e4m3,fp8_e5m2,nvfp4,int16,fp16,bf16,tf32,fp32,fp64"

# 报告展示顺序：按位宽从高到低排列（同位宽按严格度/类型分组）。
# fp64(严格 64 位) 在最前，tf32/fp32 等低位宽或近似精度往后。
# 不在此表中的精度名排到最后，并按字母序兜底。
PRECISION_ORDER = (
    "fp64", "fp32", "tf32", "bf16", "fp16", "int16",
    "int8", "fp8_e4m3", "fp8_e5m2", "nvfp4", "int4",
)


def precision_sort_key(name: str) -> tuple[int, str]:
    """精度排序键：按 PRECISION_ORDER 的位宽顺序，未列出者排末尾并按字母序。"""
    try:
        return (0, PRECISION_ORDER.index(name))
    except ValueError:
        return (1, name)


def run_command(command: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def query_gpus() -> list[dict[str, Any]]:
    fields = ["index", "name", "uuid", "memory.total", "compute_cap", "power.limit"]
    proc = run_command(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if proc.returncode != 0:
        # compute_cap is unavailable in some older drivers.
        fields = ["index", "name", "uuid", "memory.total", "power.limit"]
        proc = run_command(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ]
        )
    if proc.returncode != 0:
        raise RuntimeError(f"nvidia-smi GPU query failed: {proc.stderr.strip()}")

    gpus: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        values = [item.strip() for item in line.split(",")]
        item = dict(zip(fields, values))
        item["index"] = int(item["index"])
        gpus.append(item)
    return gpus


def query_driver() -> dict[str, str]:
    proc = run_command(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    driver = proc.stdout.splitlines()[0].strip() if proc.returncode == 0 and proc.stdout else "unknown"
    nvcc = run_command(["nvcc", "--version"])
    nvcc_line = next(
        (line.strip() for line in nvcc.stdout.splitlines() if "release" in line.lower()),
        "unknown",
    )
    return {"driver_version": driver, "nvcc_version": nvcc_line}


def parse_gpu_selection(value: str, available: list[dict[str, Any]]) -> list[int]:
    available_ids = {int(gpu["index"]) for gpu in available}
    if value.lower() == "all":
        return sorted(available_ids)
    selected = [int(item.strip()) for item in value.split(",") if item.strip()]
    invalid = sorted(set(selected) - available_ids)
    if invalid:
        raise ValueError(f"GPU IDs not present: {invalid}; available: {sorted(available_ids)}")
    if not selected:
        raise ValueError("No GPUs selected")
    return selected


def read_telemetry(device: int) -> dict[str, str]:
    fields = [
        "temperature.gpu",
        "power.draw",
        "clocks.sm",
        "clocks.mem",
        "utilization.gpu",
    ]
    proc = run_command(
        [
            "nvidia-smi",
            f"--id={device}",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    values = [item.strip() for item in proc.stdout.splitlines()[0].split(",")]
    return dict(zip(fields, values))


def run_one(
    binary: pathlib.Path,
    device: int,
    mode: str,
    output_dir: pathlib.Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_path = output_dir / f"{mode}_gpu{device}.json"
    command = [
        str(binary),
        "--device",
        str(device),
        "--output",
        str(output_path),
        "--precisions",
        args.precisions,
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
        "--workspace-mb",
        str(args.workspace_mb),
    ]
    if args.quick:
        command.append("--quick")

    telemetry_before = read_telemetry(device)
    try:
        proc = run_command(command, timeout=args.timeout)
    except subprocess.TimeoutExpired as exc:
        return {
            "mode": mode,
            "device_id": device,
            "status": "TIMEOUT",
            "error": f"Timed out after {args.timeout}s",
            "command": shlex.join(command),
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }
    telemetry_after = read_telemetry(device)

    if proc.returncode != 0:
        return {
            "mode": mode,
            "device_id": device,
            "status": "ERROR",
            "error": f"Benchmark exited with code {proc.returncode}",
            "command": shlex.join(command),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "mode": mode,
            "device_id": device,
            "status": "ERROR",
            "error": f"Cannot read benchmark JSON: {exc}",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    data["mode"] = mode
    data["status"] = "OK"
    data["telemetry_before"] = telemetry_before
    data["telemetry_after"] = telemetry_after
    return data


def best_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        mode = run.get("mode", "unknown")
        device_id = run.get("device_id", run.get("device", {}).get("id", -1))
        if run.get("status") != "OK":
            rows.append(
                {
                    "mode": mode,
                    "device_id": device_id,
                    "gpu_name": "",
                    "precision": "",
                    "status": run.get("status", "ERROR"),
                    "message": run.get("error", ""),
                }
            )
            continue
        device = run.get("device", {})
        for result in run.get("results", []):
            row = dict(result)
            row.update(
                {
                    "mode": mode,
                    "device_id": device.get("id", device_id),
                    "gpu_name": device.get("name", "unknown"),
                    "compute_capability": device.get("compute_capability", "unknown"),
                }
            )
            rows.append(row)
    return rows


def format_value(value: Any, digits: int = 2) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return ""


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "mode",
        "device_id",
        "gpu_name",
        "compute_capability",
        "precision",
        "input_type",
        "accumulator_type",
        "output_type",
        "m",
        "n",
        "k",
        "median_ms",
        "p90_ms",
        "throughput",
        "unit",
        "best",
        "validation",
        "status",
        "algorithm",
        "workspace_bytes",
        "message",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(
    path: pathlib.Path,
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("Dense GPU GEMM Benchmark")
    lines.append(f"Generated: {payload['generated_utc']}")
    lines.append(f"Driver: {payload['environment']['driver_version']}")
    lines.append(f"CUDA compiler: {payload['environment']['nvcc_version']}")
    lines.append("Dense only: no 2:4 sparsity or sparse GEMM APIs")
    lines.append("")

    best = [row for row in rows if row.get("best") is True]
    unsupported = [
        row
        for row in rows
        if row.get("status") in {"UNSUPPORTED", "SKIPPED"} and not row.get("best")
    ]
    for mode in ("isolated", "concurrent"):
        mode_rows = [row for row in best if row.get("mode") == mode]
        if not mode_rows:
            continue
        lines.append(f"[{mode.upper()}]")
        for device in sorted({int(row["device_id"]) for row in mode_rows}):
            gpu_rows = [row for row in mode_rows if int(row["device_id"]) == device]
            name = gpu_rows[0].get("gpu_name", "unknown")
            lines.append(f"GPU {device}: {name}")
            lines.append(
                "  Precision      M=N=K      Median(ms)      Throughput       Validation"
            )
            for row in sorted(
                gpu_rows, key=lambda value: precision_sort_key(str(value.get("precision", "")))
            ):
                precision = str(row.get("precision", ""))
                size = int(row.get("m", 0))
                median = format_value(row.get("median_ms"), 3)
                throughput = format_value(row.get("throughput"), 2)
                unit = str(row.get("unit", ""))
                validation = str(row.get("validation", ""))
                lines.append(
                    f"  {precision:<14} {size:<10} {median:<15} "
                    f"{throughput:>10} {unit:<8} {validation}"
                )
            lines.append("")

        if mode == "concurrent":
            lines.append("  Concurrent aggregate:")
            for precision in sorted(
                {str(row.get("precision")) for row in mode_rows}, key=precision_sort_key
            ):
                precision_rows = [
                    row for row in mode_rows if row.get("precision") == precision
                ]
                units = {row.get("unit") for row in precision_rows}
                if len(units) != 1:
                    continue
                total = sum(float(row.get("throughput", 0.0)) for row in precision_rows)
                lines.append(
                    f"    {precision:<14} {total:>10.2f} {next(iter(units))}"
                )
            lines.append("")

    if unsupported:
        lines.append("[UNSUPPORTED / SKIPPED]")
        seen: set[tuple[Any, ...]] = set()
        for row in unsupported:
            key = (
                row.get("mode"),
                row.get("device_id"),
                row.get("precision"),
                row.get("message"),
            )
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"  {row.get('mode')} GPU {row.get('device_id')} "
                f"{row.get('precision')}: {row.get('status')} - {row.get('message', '')}"
            )
        lines.append("")

    errors = [
        row
        for row in rows
        if row.get("status") in {"ERROR", "TIMEOUT", "VALIDATION_FAILED"}
    ]
    if errors:
        lines.append("[ERRORS]")
        for row in errors:
            lines.append(
                f"  {row.get('mode')} GPU {row.get('device_id')}: "
                f"{row.get('status')} - {row.get('message', '')}"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run dense GPU GEMM benchmarks on one or more NVIDIA GPUs."
    )
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--gpus", default="all", help="all or a comma-separated GPU list")
    parser.add_argument(
        "--precisions",
        default=DEFAULT_PRECISIONS,
        help="Comma-separated precision list",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "isolated", "concurrent"),
        default="both",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--workspace-mb", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Result directory; default: results/<UTC timestamp>",
    )
    args = parser.parse_args()

    if not args.binary.is_file():
        parser.error(f"benchmark binary not found: {args.binary}")
    if args.iters < 1 or args.warmup < 0 or args.workspace_mb < 0:
        parser.error("iters must be positive; warmup/workspace must be non-negative")

    gpus = query_gpus()
    selected = parse_gpu_selection(args.gpus, gpus)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output or pathlib.Path(__file__).resolve().parents[1] / "results" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    if args.mode in {"both", "isolated"}:
        print("Running isolated per-GPU measurements...")
        for device in selected:
            print(f"  GPU {device}")
            runs.append(run_one(args.binary, device, "isolated", output_dir, args))

    if args.mode in {"both", "concurrent"}:
        print("Running concurrent all-GPU measurements...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected)) as pool:
            futures = {
                pool.submit(
                    run_one,
                    args.binary,
                    device,
                    "concurrent",
                    output_dir,
                    args,
                ): device
                for device in selected
            }
            for future in concurrent.futures.as_completed(futures):
                device = futures[future]
                try:
                    runs.append(future.result())
                    print(f"  GPU {device} completed")
                except Exception as exc:  # defensive isolation of worker failures
                    runs.append(
                        {
                            "mode": "concurrent",
                            "device_id": device,
                            "status": "ERROR",
                            "error": f"Runner exception: {exc}",
                        }
                    )

    rows = best_rows(runs)
    payload = {
        "schema_version": 1,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": query_driver(),
        "selected_gpus": selected,
        "gpu_inventory": gpus,
        "configuration": {
            "mode": args.mode,
            "quick": args.quick,
            "precisions": args.precisions.split(","),
            "iters": args.iters,
            "warmup": args.warmup,
            "workspace_mb": args.workspace_mb,
            "timeout_seconds": args.timeout,
            "sparsity": "none",
        },
        "runs": runs,
    }
    json_path = output_dir / "report.json"
    csv_path = output_dir / "report.csv"
    md_path = output_dir / "report.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(csv_path, rows)
    write_text(md_path, payload, rows)

    print(f"Reports written to: {output_dir}")
    print(f"  {md_path.name}")
    print(f"  {csv_path.name}")
    print(f"  {json_path.name}")
    failed = any(
        row.get("status") in {"ERROR", "TIMEOUT", "VALIDATION_FAILED"}
        for row in rows
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
