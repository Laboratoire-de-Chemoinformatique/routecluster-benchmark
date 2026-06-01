#!/usr/bin/env python3
"""Run the notebook-equivalent AiZynthFinder search and export unique routes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aizynthfinder.aizynthfinder import AiZynthFinder

from cgr_clustering.aizynth_converter import (
    extract_routes_from_tree,
    filter_unique_routes,
)
from route_data import write_export

APATINIB = "N#CC1(c2ccc(NC(=O)c3cccnc3NCc3ccncc3)cc2)CCCC1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--output", default="benchmark_results/aizynth_routes.json")
    parser.add_argument("--target", default=APATINIB)
    parser.add_argument("--stocks", nargs="*", help="defaults to every loaded stock")
    parser.add_argument("--expansion-policies", nargs="+", default=["uspto"])
    parser.add_argument("--filter-policies", nargs="*", default=[])
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--time-limit", type=float, help="search time limit in seconds")
    parser.add_argument("--max-transforms", type=int)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    finder = AiZynthFinder(configfile=args.config)

    selected_stocks = args.stocks or list(finder.stock.items)
    finder.stock.select(selected_stocks)
    finder.expansion_policy.select(args.expansion_policies)
    if args.filter_policies:
        finder.filter_policy.select(args.filter_policies)
    else:
        finder.filter_policy.deselect()

    if args.iterations is not None:
        finder.config.search.iteration_limit = args.iterations
    if args.time_limit is not None:
        finder.config.search.time_limit = args.time_limit
    if args.max_transforms is not None:
        finder.config.search.max_transforms = args.max_transforms

    finder.target_smiles = args.target
    finder.prepare_tree()
    search_seconds = finder.tree_search(show_progress=not args.no_progress)

    route_collections = extract_routes_from_tree(SimpleNamespace(finder=finder))
    unique_routes = filter_unique_routes(route_collections)
    metadata = {
        "target": args.target,
        "config": args.config,
        "stocks": selected_stocks,
        "expansion_policies": args.expansion_policies,
        "filter_policies": args.filter_policies,
        "iteration_limit": finder.config.search.iteration_limit,
        "time_limit_seconds": finder.config.search.time_limit,
        "max_transforms": finder.config.search.max_transforms,
        "search_seconds": search_seconds,
        "search_stats": finder.search_stats,
        "solved_route_count": len(route_collections),
        "unique_route_count": len(unique_routes),
    }
    write_export(args.output, routes=list(unique_routes.dicts), metadata=metadata)
    print(f"Exported {len(unique_routes)} unique routes to {args.output}")
    print(f"Solved route leaves before deduplication: {len(route_collections)}")


if __name__ == "__main__":
    main()
