#!/usr/bin/env bash
#
# refinetune.sh -- carry a trained model onto a new game build.
#
# Run after scripts/on_update.sh has refreshed the decompile (and after
# scripts/sync_content.py --write if the patch added content). A patch changes
# card values, not the shape of the game, so the previous model's structure is
# still broadly right and only needs re-fitting to the new numbers. That is far
# cheaper than starting over and usually better.
#
# It picks the newest model in the output tree, checks it, and fine-tunes into a
# NEW directory. The source model is never overwritten -- if the fine-tune goes
# badly you still have the thing that worked.
#
# WHAT MAKES THIS SAFE, AND WHERE IT ISN'T
#
# Observation and action shapes do not depend on how much content exists:
# OBS_SIZE is built from MAX_HAND_SIZE and MAX_ENEMIES, and the action space is a
# fixed 115 (combat) or 157 (full run). So adding cards cannot invalidate a
# checkpoint, and MaskablePPO.load raises loudly if the spaces ever do disagree.
#
# It is not a perfectly clean resume, though. A card's identity is encoded as its
# ordinal in CardId divided by the total count, so adding enum members shifts
# every card's encoded value slightly -- Strike goes from 11/601 to 11/624. Small,
# and fine-tuning re-adapts, but it is why this runs for a real budget rather than
# a token one.
#
# Usage:
#   scripts/refinetune.sh combat            # fine-tune the newest combat model
#   scripts/refinetune.sh run               # fine-tune the newest full-run model
#   scripts/refinetune.sh run --from output/run_ppo_v1/final_model.zip
#   scripts/refinetune.sh run --steps 2000000
#
# Exit codes: 0 done, 2 could not run.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO/output}"

die() { echo "ERROR: $*" >&2; exit 2; }

MODE="${1:-}"
shift || true
case "$MODE" in
    combat) TRAINER="train_combat.py";   PREFIX="combat_ppo"; DEFAULT_STEPS=500000 ;;
    run)    TRAINER="train_full_run.py"; PREFIX="run_ppo";    DEFAULT_STEPS=2000000 ;;
    -h|--help|"") awk 'NR>1 && /^#/{print substr($0,3); next} NR>1{exit}' "$0"; exit 0 ;;
    *)      die "unknown mode '$MODE' (expected 'combat' or 'run')" ;;
esac

FROM=""
STEPS="$DEFAULT_STEPS"
EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --from)  FROM="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        *)       EXTRA+=("$1"); shift ;;
    esac
done

[ -x "$PYTHON" ] || die "python not found at $PYTHON (set PYTHON=)"

# Newest final_model.zip under a matching output directory. final_model rather
# than best_model deliberately: best_model is chosen by the evaluation callback,
# and with a small n_eval_episodes that choice is mostly noise.
if [ -z "$FROM" ]; then
    FROM="$(find "$OUTPUT_ROOT" -maxdepth 2 -path "*${PREFIX}*" -name 'final_model.zip' \
            -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)"
fi
[ -n "$FROM" ] && [ -f "$FROM" ] || die "no model to fine-tune from under $OUTPUT_ROOT (pass --from)"

SOURCE_DIR="$(dirname "$FROM")"
echo "=== Fine-tuning $MODE ==="
echo "  from:  $FROM"

# What build did that model learn? Not a gate -- fine-tuning onto a new build is
# the entire point -- but it should be stated, because a model carried forward
# across several patches without anyone noticing is exactly the drift this
# project keeps finding.
if [ -f "$SOURCE_DIR/game_build.json" ]; then
    "$PYTHON" - "$SOURCE_DIR/game_build.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  trained on build: {d.get('dll_sha256','?')[:16]}  ({d.get('informational_version','?')})")
PY
else
    echo "  trained on build: unknown (no game_build.json -- predates the stamp)"
fi

"$PYTHON" - <<'PY'
from sts2_env.core.game_build import check_decompile_matches_installed
ok, reason = check_decompile_matches_installed()
print(f"  installed build:  {reason}")
PY

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$OUTPUT_ROOT/${PREFIX}_ft_${STAMP}"
echo "  into:  $OUT"
echo "  steps: $STEPS"
echo

# The trainer's own build guard still applies: it refuses to run against a
# decompile that is not the installed game, which is the whole reason to trust
# what comes out of this.
exec "$PYTHON" "$REPO/scripts/$TRAINER" \
    --resume-from "$FROM" \
    --total-timesteps "$STEPS" \
    --output-dir "$OUT" \
    "${EXTRA[@]}"
