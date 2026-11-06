"""Published numbers transcribed from the papers, for reference lines only.

Nothing in this module is a measurement of ours. It exists so a figure can show
where the literature sits without us re-running someone else's experiment, and
so that number can never be confused with one of our runs: every series here is
drawn in neutral grey dashes (``style.PUBLISHED_STYLE``), every label carries
the word "published", and every helper returns a payload tagged
``published=True``.

Scope warning, and it is a narrow one: these tables are reported on the MRL
fixed-feature ResNet-50 2048-d backbone, which is our ``ff2048`` and nothing
else. ``rn18``, ``convnext_tiny``, ``vit_b16`` and ``dinov2_vits14`` have no
comparable published counterpart -- CSR Table 9 studies input dimension under a
different protocol and CSRv2 reports no image numbers for them -- so asking for
a published curve on those backbones returns ``None`` rather than a plausible
line.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

#: The only backbone for which published numbers are comparable to ours.
PUBLISHED_BACKBONE = "ff2048"

# CSR, arXiv 2503.01776, Table 4 ("ImageNet-1k, 1-NN accuracy"), row CSR,
# backbone = MRL fixed-feature ResNet-50 FF2048. The table's "active dimension"
# is the Top-K budget, so the key is K.
_CSR_TABLE4_CSR: Mapping[int, float] = MappingProxyType({
    2: 61.05, 4: 65.33, 8: 67.78, 16: 69.17, 32: 70.15, 64: 70.94,
})

# Same table, row MRL. Matryoshka is a *published reference only*: we never
# retrain it, so it must never enter a paired comparison or a delta figure.
_CSR_TABLE4_MRL: Mapping[int, float] = MappingProxyType({
    2: 47.81, 4: 55.65, 8: 62.19, 16: 67.91, 32: 69.46, 64: 70.17,
})

# CSRv2, arXiv 2602.05735, Table 12 / Fig. 5(a), same FF2048 backbone. The
# released CSRv2 ships no image checkpoint, so "CSRv2-linear" (linear
# k-annealing, which is what our ``csrv2`` method implements) is the comparable
# variant; the paper's remaining variants are not reproducible from the release.
_CSRV2_TABLE12_LINEAR: Mapping[int, float] = MappingProxyType({
    2: 65.78, 4: 67.29, 8: 68.42,
})

# CSR Table 4, "Full Rep." column: 1-NN accuracy of the uncompressed 2048-d
# embedding. This doubles as acceptance gate G1 for our own extraction
# (``config.BACKBONES["ff2048"].reference_dense_top1``); the two agreeing is a
# check on our pipeline, not a result of ours.
_DENSE_TOP1: Mapping[str, float] = MappingProxyType({"ff2048": 71.19})


#: Series name -> (curve, pretty label, provenance string).
_SERIES: Mapping[str, tuple[Mapping[int, float], str, str]] = MappingProxyType({
    "csr": (_CSR_TABLE4_CSR, "CSR (published)", "CSR arXiv 2503.01776, Table 4"),
    "mrl": (_CSR_TABLE4_MRL, "MRL (published)", "CSR arXiv 2503.01776, Table 4"),
    "csrv2_linear": (_CSRV2_TABLE12_LINEAR, "CSRv2-linear (published)",
                     "CSRv2 arXiv 2602.05735, Table 12 / Fig. 5(a)"),
})

#: Draw order for overlays; coarse-to-fine so the strongest baseline is last.
PUBLISHED_SERIES = ("mrl", "csr", "csrv2_linear")


def published_curve(method: str, backbone: str) -> dict[int, float] | None:
    """``{K: top1}`` for a published series, or ``None`` when none exists.

    ``method`` names a *published series* (see ``PUBLISHED_SERIES``), not one of
    our trained methods. The names deliberately overlap with ``config.METHODS``
    for ``csr`` only because it is the same algorithm as published; the returned
    numbers are still theirs, not ours, and the caller must label them so.
    """
    if backbone != PUBLISHED_BACKBONE:
        return None
    entry = _SERIES.get(method)
    if entry is None:
        return None
    return dict(entry[0])


def published_label(method: str) -> str | None:
    """Pretty, unmistakably-published label for a series."""
    entry = _SERIES.get(method)
    return None if entry is None else entry[1]


def published_provenance(method: str) -> str | None:
    """Paper, table and figure a series was transcribed from."""
    entry = _SERIES.get(method)
    return None if entry is None else entry[2]


def dense_reference(backbone: str) -> dict[str, object] | None:
    """Published dense ("Full Rep.") 1-NN top-1, or ``None`` if unpublished.

    Returns a payload rather than a bare float so a caller cannot drop the
    provenance on the way to a figure caption.
    """
    top1 = _DENSE_TOP1.get(backbone)
    if top1 is None:
        return None
    return {"backbone": backbone, "top1": top1, "published": True,
            "label": "Full Rep. (published)",
            "provenance": "CSR arXiv 2503.01776, Table 4, 'Full Rep.' column"}


def overlays(backbone: str) -> list[dict[str, object]]:
    """Every published overlay available for ``backbone``, ready to plot.

    Each entry is ``{series, label, provenance, curve, published}``. Empty for
    every backbone except ``ff2048``, which is the honest answer: we have no
    published number to compare against on the other four.
    """
    out: list[dict[str, object]] = []
    for name in PUBLISHED_SERIES:
        curve = published_curve(name, backbone)
        if not curve:
            continue
        out.append({"series": name, "label": published_label(name),
                    "provenance": published_provenance(name),
                    "curve": curve, "published": True})
    return out
