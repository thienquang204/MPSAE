"""Raw ImageNet-1K: folders for ``ImageFolder`` and an FFCV file for CSRv2.

torch, torchvision and ffcv are imported inside the functions, so planning and
reporting code can import this package on a machine without CUDA.
"""
from __future__ import annotations

from .ffcv_write import write_ffcv
from .imagenet import imagenet_folder_stats, prepare_imagenet

__all__ = ["imagenet_folder_stats", "prepare_imagenet", "write_ffcv"]
