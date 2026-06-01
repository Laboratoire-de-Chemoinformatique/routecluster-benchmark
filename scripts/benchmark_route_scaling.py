#!/usr/bin/env python3
"""Measure wall-clock scaling for SB-CGR clustering, AB, and TED matrices."""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from route_data import load_routes

METHODS = ("sb-cgr", "ab", "ted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("routes", help="JSON route export from export_aizynth_routes.py")
    parser.add_argument(
        "--counts",
        type=int,
        nargs="+",
        help="defaults to 25, 50, 100, 150, 200, and the full exported route count",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seed", type=int, default=20250531)
    parser.add_argument("--sampling", choices=("random", "prefix"), default="random")
    parser.add_argument("--ted-content", choices=("both", "molecules", "reactions"), default="both")
    parser.add_argument("--ted-timeout", type=int)
    parser.add_argument("--output-dir", default="benchmark_results/route_scaling")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def timed(function):
    start = perf_counter()
    result = function()
    return perf_counter() - start, result


def benchmark_sb_cgr(routes: list[dict]) -> list[tuple[str, float, str]]:
    from synplan.chem.reaction_routes.clustering import cluster_routes

    from cgr_clustering.aizynth_converter import extract_all_route_cgrs
    from cgr_clustering.sb_clustering import compose_all_sb_cgrs

    records = [{"dict": route} for route in routes]
    total_start = perf_counter()
    extract_seconds, route_cgrs = timed(
        lambda: extract_all_route_cgrs(
            records,
            check_trans_error=True,
            show_progress=False,
        )
    )
    reduce_seconds, sb_cgrs = timed(lambda: compose_all_sb_cgrs(route_cgrs))
    cluster_seconds, clusters = timed(lambda: cluster_routes(sb_cgrs))
    total_seconds = perf_counter() - total_start
    cluster_count = str(len(clusters))
    return [
        ("sb-cgr-extract", extract_seconds, cluster_count),
        ("sb-cgr-reduce", reduce_seconds, cluster_count),
        ("sb-cgr-cluster", cluster_seconds, cluster_count),
        ("sb-cgr-total", total_seconds, cluster_count),
    ]


def benchmark_ab(routes: list[dict]) -> list[tuple[str, float, str]]:
    from rxnutils.routes.comparison import simple_route_similarity
    from rxnutils.routes.readers import read_aizynthfinder_dict

    def compute():
        parsed_routes = [read_aizynthfinder_dict(route) for route in routes]
        return 1.0 - simple_route_similarity(parsed_routes)

    seconds, matrix = timed(compute)
    assert matrix.shape == (len(routes), len(routes))
    return [("ab-matrix", seconds, "")]


def benchmark_ted(
    routes: list[dict],
    *,
    content: str,
    timeout: int | None,
) -> list[tuple[str, float, str]]:
    from route_distances.ted.distances import distance_matrix

    seconds, matrix = timed(lambda: distance_matrix(routes, content=content, timeout=timeout))
    assert matrix.shape == (len(routes), len(routes))
    return [("ted-matrix", seconds, "")]


def select_routes(
    routes: list[dict],
    count: int,
    *,
    sampling: str,
    seed: int,
) -> list[dict]:
    if sampling == "prefix":
        return routes[:count]
    return random.Random(seed).sample(routes, count)


def write_csv(filename: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with filename.open("w", newline="") as fileobj:
        writer = csv.DictWriter(fileobj, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(raw_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in raw_rows:
        grouped[(row["method"], row["route_count"])].append(float(row["seconds"]))

    summary_rows = []
    for (method, route_count), values in sorted(grouped.items()):
        summary_rows.append(
            {
                "method": method,
                "route_count": route_count,
                "repeats": len(values),
                "mean_seconds": statistics.mean(values),
                "median_seconds": statistics.median(values),
                "stdev_seconds": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min_seconds": min(values),
                "max_seconds": max(values),
            }
        )
    return summary_rows


def scaling_fits(summary_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[row["method"]].append(row)

    fit_rows = []
    for method, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["route_count"]))
        if len(rows) < 2:
            continue
        counts = np.asarray([int(row["route_count"]) for row in rows], dtype=float)
        seconds = np.asarray([float(row["median_seconds"]) for row in rows], dtype=float)
        slope, intercept = np.polyfit(np.log(counts), np.log(seconds), deg=1)
        prediction = slope * np.log(counts) + intercept
        residual = np.square(np.log(seconds) - prediction).sum()
        total = np.square(np.log(seconds) - np.log(seconds).mean()).sum()
        fit_rows.append(
            {
                "method": method,
                "log_log_slope": slope,
                "log_log_intercept": intercept,
                "r_squared": 1.0 - residual / total if total else 1.0,
            }
        )
    return fit_rows


def plot_summary(summary_rows: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[row["method"]].append(row)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for method, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["route_count"]))
        counts = [int(row["route_count"]) for row in rows]
        medians = [float(row["median_seconds"]) for row in rows]
        mins = [float(row["min_seconds"]) for row in rows]
        maxes = [float(row["max_seconds"]) for row in rows]
        for axis in axes:
            axis.plot(counts, medians, marker="o", label=method)
            axis.fill_between(counts, mins, maxes, alpha=0.15)
    axes[0].set_xlabel("Route count")
    axes[0].set_ylabel("Wall-clock time (s)")
    axes[1].set_xlabel("Route count")
    axes[1].set_ylabel("Wall-clock time (s)")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)


def main() -> None:
    args = parse_args()
    routes = load_routes(args.routes)
    default_counts = [count for count in [25, 50, 100, 150, 200] if count < len(routes)]
    counts = sorted(set(args.counts or [*default_counts, len(routes)]))
    if not counts or counts[0] < 1 or counts[-1] > len(routes):
        raise SystemExit(f"Counts must be between 1 and {len(routes)}")
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")

    raw_rows = []
    for count in counts:
        for repeat in range(args.repeats):
            sample_seed = args.seed + count * 1000 + repeat
            selected_routes = select_routes(
                routes,
                count,
                sampling=args.sampling,
                seed=sample_seed,
            )
            for method in args.methods:
                if method == "sb-cgr":
                    results = benchmark_sb_cgr(selected_routes)
                elif method == "ab":
                    results = benchmark_ab(selected_routes)
                else:
                    results = benchmark_ted(
                        selected_routes,
                        content=args.ted_content,
                        timeout=args.ted_timeout,
                    )
                for measurement, seconds, cluster_count in results:
                    print(
                        f"routes={count} repeat={repeat + 1}/{args.repeats} "
                        f"method={measurement} seconds={seconds:.6f}"
                    )
                    raw_rows.append(
                        {
                            "route_count": count,
                            "repeat": repeat + 1,
                            "sample_seed": sample_seed,
                            "sampling": args.sampling,
                            "method": measurement,
                            "seconds": seconds,
                            "cluster_count": cluster_count,
                        }
                    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize(raw_rows)
    fit_rows = scaling_fits(summary_rows)
    write_csv(
        output_dir / "timings_raw.csv",
        raw_rows,
        ["route_count", "repeat", "sample_seed", "sampling", "method", "seconds", "cluster_count"],
    )
    write_csv(
        output_dir / "timings_summary.csv",
        summary_rows,
        [
            "method",
            "route_count",
            "repeats",
            "mean_seconds",
            "median_seconds",
            "stdev_seconds",
            "min_seconds",
            "max_seconds",
        ],
    )
    write_csv(
        output_dir / "scaling_fits.csv",
        fit_rows,
        ["method", "log_log_slope", "log_log_intercept", "r_squared"],
    )
    if not args.no_plot:
        plot_summary(summary_rows, output_dir / "wall_clock_scaling.png")
    print(f"Wrote timing data to {output_dir}")


if __name__ == "__main__":
    main()
