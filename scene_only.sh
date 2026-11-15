#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf 'Usage: retired launcher. Use python -m mpsae.patch_scene with configs/patch_scene_voc.yaml or configs/patch_scene_nyuv2.yaml. See docs/patch_scene_kaggle.md.\n'
    exit 0
fi
printf 'The old checkpoint-adaptation segmentation workflow is retired. Use the VOC/NYUv2 patch-SAE launcher in docs/patch_scene_kaggle.md. No jobs were started.\n' >&2
exit 2
