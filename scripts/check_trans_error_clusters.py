#!/usr/bin/env python3
"""Check the effect of transamidation correction on SB-CGR route clusters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from synplan.chem.reaction_routes.clustering import cluster_routes

from cgr_clustering.aizynth_converter import extract_all_route_cgrs
from cgr_clustering.sb_clustering import compose_all_sb_cgrs
from route_data import load_routes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("routes", help="JSON route export from export_aizynth_routes.py")
    parser.add_argument("--expect-route-count", type=int)
    parser.add_argument("--expect-clusters", type=int)
    parser.add_argument("--compare-disabled", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def cluster(routes: list[dict], *, check_trans_error: bool, show_progress: bool):
    records = [{"dict": route} for route in routes]
    route_cgrs = extract_all_route_cgrs(
        records,
        check_trans_error=check_trans_error,
        show_progress=show_progress,
    )
    sb_cgrs = compose_all_sb_cgrs(route_cgrs)
    clusters = cluster_routes(sb_cgrs)
    return route_cgrs, sb_cgrs, clusters


def changed_ids(left: dict, right: dict) -> list[int]:
    return [route_id for route_id in left if str(left[route_id]) != str(right[route_id])]


def main() -> None:
    args = parse_args()
    routes = load_routes(args.routes)
    show_progress = not args.no_progress
    print(f"Loaded routes: {len(routes)}")

    enabled_route_cgrs, enabled_sb_cgrs, enabled_clusters = cluster(
        routes,
        check_trans_error=True,
        show_progress=show_progress,
    )
    print(f"Clusters with check_trans_error=True: {len(enabled_clusters)}")

    if args.compare_disabled:
        disabled_route_cgrs, disabled_sb_cgrs, disabled_clusters = cluster(
            routes,
            check_trans_error=False,
            show_progress=show_progress,
        )
        print(f"Clusters with check_trans_error=False: {len(disabled_clusters)}")
        print(f"Changed RouteCGR route IDs: {changed_ids(enabled_route_cgrs, disabled_route_cgrs)}")
        print(f"Changed SB-CGR route IDs: {changed_ids(enabled_sb_cgrs, disabled_sb_cgrs)}")

    if args.expect_route_count is not None and len(routes) != args.expect_route_count:
        raise SystemExit(
            f"Expected {args.expect_route_count} routes, observed {len(routes)}"
        )
    if args.expect_clusters is not None and len(enabled_clusters) != args.expect_clusters:
        raise SystemExit(
            f"Expected {args.expect_clusters} enabled clusters, "
            f"observed {len(enabled_clusters)}"
        )


if __name__ == "__main__":
    main()
