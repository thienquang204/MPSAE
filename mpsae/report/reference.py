"""Retired published-overlay API, retained only for import compatibility.

Plots and report tables use experiment measurements. Historical callers may
still import these helpers, but all lookups are empty and this module contains
no external result values. It is not imported by the report package.
"""
from __future__ import annotations

PUBLISHED_BACKBONE = None
PUBLISHED_SERIES: tuple[str, ...] = ()


def published_curve(method: str, backbone: str) -> dict[int, float] | None:
    return None


def published_label(method: str) -> str | None:
    return None


def published_provenance(method: str) -> str | None:
    return None


def dense_reference(backbone: str) -> dict[str, object] | None:
    return None


def overlays(backbone: str) -> list[dict[str, object]]:
    return []
