"""Helpers for portable AiZynthFinder route exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_routes(filename: str | Path) -> list[dict[str, Any]]:
    """Load either a route list or an export payload containing a route list."""
    path = Path(filename)
    with path.open() as fileobj:
        payload = json.load(fileobj)

    routes = payload.get("routes") if isinstance(payload, dict) else payload
    if not isinstance(routes, list):
        raise ValueError(f"{path} must contain a route list or a 'routes' list")
    if not routes:
        raise ValueError(f"{path} contains no routes")
    if not all(isinstance(route, dict) for route in routes):
        raise ValueError(f"{path} contains a non-object route")
    return routes


def write_export(
    filename: str | Path,
    *,
    routes: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    """Write routes and search metadata in a portable JSON payload."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fileobj:
        json.dump({"metadata": metadata, "routes": routes}, fileobj, indent=2)
        fileobj.write("\n")
