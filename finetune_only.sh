#!/usr/bin/env bash
# Additional backbone fine-tuning was removed for resource reasons.
# Exit before setup, downloads, model loading or remote writes.
set -Eeuo pipefail
printf '%s\n' \
  'Disabled: all nine additional backbone fine-tuning experiments and their evaluations.' \
  'The main frozen-backbone experiment settings and saved artifacts are unchanged.' \
  'Historical recipe: configs/archive/finetune_gradient_replay.yaml'
