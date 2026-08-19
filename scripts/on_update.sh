#!/usr/bin/env bash
#
# on_update.sh -- run this after Slay the Spire 2 updates.
#
# The simulator is a hand-written reimplementation of the game, so a patch breaks
# it silently: no compile error, no exception, just numbers that quietly stop
# matching and a policy that trains confidently against a game that no longer
# exists. The mod fails loudly; the simulator does not. This script is the noise.
#
# It does seven things, in order:
#
#   1. Notices whether the installed build actually changed.
#   2. Decompiles it, keeping the previous decompile so there is something to
#      compare against.
#   3. Diffs the two game builds -- the short list of what this patch did.
#      Also writes an exhaustive changes.txt with every added, removed, and
#      changed class for sharing or review.
#   4. Diffs card values against the simulator.
#   5. Diffs content inventory (cards, relics, monsters, ...) against the simulator.
#   6. Reports content the simulator has no name for at all -- the enum members
#      scripts/sync_content.py --write would add.
#   7. Diffs MONSTER constants against the simulator. Cards are covered by step
#      4 and by derived_values.py building them from the decompile; monsters are
#      hand-copied, and every parity bug found in the week of 2026-08-19 was one
#      -- including an Axebot at 40-44 HP against the game's 70-78, which cost
#      51 HP of 93 in a single fight of the deepest run on record.
#
# What it deliberately does NOT do is edit sts2_env/ for you. Auto-applying the
# value diffs was the obvious next step and it is a trap: the extractor reads a
# card's first DamageVar, which is right for most cards and wrong for the ones
# with conditional or multi-hit damage, and a script that "fixed" those would
# write wrong numbers into the simulator with no diff left to review. Reporting a
# suspect number costs a minute of reading; silently writing one costs a training
# run. So this narrows 596 cards down to the handful you must look at, and you
# make the edit.
#
# Usage:
#   scripts/on_update.sh              # skip everything if the build is unchanged
#   scripts/on_update.sh --force      # re-decompile regardless
#   scripts/on_update.sh --no-decompile   # just re-run the reports
#
# Overridable: STS2_DIR, DECOMPILE_DIR, REPORT_DIR, PYTHON, DOTNET_ROOT
#
# Exit codes: 0 nothing to do, 1 drift found (act on it), 2 could not run.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STS2_DIR="${STS2_DIR:-$HOME/.local/share/Steam/steamapps/common/Slay the Spire 2}"
DECOMPILE_DIR="${DECOMPILE_DIR:-/media/Bucket_Drive/development/cyra/cyra_game/decompile}"
REPORT_DIR="${REPORT_DIR:-$REPO/reports}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"

# ilspycmd targets .NET 8 and the machine has 9; without roll-forward it refuses
# to start with a bare "you must install .NET" that names no version.
export DOTNET_ROOT="${DOTNET_ROOT:-$HOME/.dotnet}"
export DOTNET_ROLL_FORWARD="${DOTNET_ROLL_FORWARD:-Major}"
PATH="$DOTNET_ROOT:$DOTNET_ROOT/tools:$PATH"
export PATH

FORCE=0
DO_DECOMPILE=1
for arg in "$@"; do
    case "$arg" in
        --force)        FORCE=1 ;;
        --no-decompile) DO_DECOMPILE=0 ;;
        # The header comment block only -- stop at the first line that is not a
        # comment, so inline comments further down do not leak into --help.
        -h|--help)      awk 'NR>1 && /^#/{print substr($0,3); next} NR>1{exit}' "$0"; exit 0 ;;
        *)              echo "unknown option: $arg" >&2 ; exit 2 ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 2; }
step() { echo; echo "=== $* ==="; }

command -v ilspycmd >/dev/null || die "ilspycmd not on PATH (DOTNET_ROOT=$DOTNET_ROOT)"
[ -x "$PYTHON" ] || die "python not found at $PYTHON (set PYTHON=)"
[ -d "$STS2_DIR" ] || die "game not found at $STS2_DIR (set STS2_DIR=)"

# One glob rather than a per-platform case: the data dir is named for the
# platform it shipped for, and there is exactly one.
DATA_DIR=""
for candidate in "$STS2_DIR"/data_sts2_*; do
    [ -d "$candidate" ] && DATA_DIR="$candidate"
done
[ -n "$DATA_DIR" ] || die "no data_sts2_* directory under $STS2_DIR"
DLL="$DATA_DIR/sts2.dll"
[ -f "$DLL" ] || die "sts2.dll not found at $DLL"

# ---------------------------------------------------------------------------
step "1/7  Installed build"
# ---------------------------------------------------------------------------

DLL_HASH="$(sha256sum "$DLL" | cut -d' ' -f1)"
STAMP="$DECOMPILE_DIR/.source.json"
PREV_HASH=""
[ -f "$STAMP" ] && PREV_HASH="$("$PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("dll_sha256",""))' "$STAMP" 2>/dev/null || true)"

echo "  dll:  $DLL"
echo "  hash: ${DLL_HASH:0:16}"
if [ -n "$PREV_HASH" ]; then
    echo "  last: ${PREV_HASH:0:16}"
fi

if [ "$DLL_HASH" = "$PREV_HASH" ] && [ "$FORCE" -eq 0 ] && [ "$DO_DECOMPILE" -eq 1 ]; then
    echo
    echo "Build unchanged since the last decompile. Nothing to do."
    echo "(--force to re-decompile anyway, --no-decompile to just re-run the reports.)"
    exit 0
fi

# ---------------------------------------------------------------------------
step "2/7  Decompile"
# ---------------------------------------------------------------------------

PREV_DIR="$DECOMPILE_DIR.prev"

if [ "$DO_DECOMPILE" -eq 1 ]; then
    STAGE="$DECOMPILE_DIR.new"
    rm -rf "$STAGE"
    mkdir -p "$STAGE"

    # Decompile into a staging directory and only rotate on success. A decompile
    # that dies halfway must not leave a truncated tree where the good one was --
    # every report downstream would read it as "the patch deleted 900 classes".
    echo "  decompiling to $STAGE ..."
    ilspycmd -p -o "$STAGE" "$DLL" >/dev/null || die "ilspycmd failed"

    NEW_COUNT="$(find "$STAGE" -name '*.cs' | wc -l)"
    [ "$NEW_COUNT" -gt 100 ] || die "decompile produced only $NEW_COUNT files; refusing to rotate"

    INFO_VERSION="$(grep -ho 'AssemblyInformationalVersion("[^"]*")' \
        "$STAGE/Properties/AssemblyInfo.cs" 2>/dev/null | head -1 | cut -d'"' -f2 || true)"

    # Only rotate a tree whose build we can name. An unstamped decompile is of
    # unknown vintage -- on the first run it is usually the same build we just
    # decompiled, and promoting it to "previous" would silently produce an empty
    # patch diff while destroying the only real baseline on disk.
    if [ -f "$STAMP" ] && [ -d "$DECOMPILE_DIR" ]; then
        rm -rf "$PREV_DIR"
        mv "$DECOMPILE_DIR" "$PREV_DIR"
    else
        [ -d "$DECOMPILE_DIR" ] && rm -rf "$DECOMPILE_DIR"
    fi
    mv "$STAGE" "$DECOMPILE_DIR"

    "$PYTHON" - "$STAMP" "$DLL_HASH" "$DLL" "${INFO_VERSION:-unknown}" <<'PY'
import json, sys, datetime
stamp, dll_hash, dll, version = sys.argv[1:5]
json.dump({"dll_sha256": dll_hash, "dll": dll, "informational_version": version,
           "decompiled_at": datetime.datetime.now().astimezone().isoformat()},
          open(stamp, "w"), indent=2)
PY
    echo "  $NEW_COUNT files, version ${INFO_VERSION:-unknown}"
    [ -d "$PREV_DIR" ] && echo "  previous decompile kept at $PREV_DIR"
else
    echo "  skipped (--no-decompile)"
fi

[ -d "$DECOMPILE_DIR" ] || die "no decompile at $DECOMPILE_DIR"

VERSION_TAG="$("$PYTHON" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("informational_version","unknown").split("+")[0] + "-" + d.get("dll_sha256","")[:8])' \
    "$STAMP" 2>/dev/null || echo "unknown")"
OUT="$REPORT_DIR/$VERSION_TAG"
mkdir -p "$OUT"

drift=0

# ---------------------------------------------------------------------------
step "3/7  What the patch changed (game vs game)"
# ---------------------------------------------------------------------------

# `.prev` is the only real baseline. The fallback below used to be described as
# "the committed decompiled/ tree", and it is not committed and it is not a tree:
# `decompiled` is a SYMLINK, tracked in git as mode 120000, pointing at
# $DECOMPILE_DIR itself. Using it as a baseline diffs the tree just written
# against itself and reports no changes -- which is the most reassuring possible
# way to be wrong, and would have fired on any machine without a .prev.
#
# So it is only used when it genuinely resolves somewhere else, and the script
# says plainly when it has nothing to compare against.
BASELINE=""
if [ -d "$PREV_DIR" ]; then
    BASELINE="$PREV_DIR"
    echo "  baseline: previous decompile ($PREV_DIR)"
elif [ -d "$REPO/decompiled" ] \
     && [ "$(readlink -f "$REPO/decompiled")" != "$(readlink -f "$DECOMPILE_DIR")" ]; then
    BASELINE="$REPO/decompiled"
    echo "  baseline: $REPO/decompiled -- a different tree, but an older one, so this"
    echo "            diff spans however many builds have landed since it was taken."
elif [ -d "$REPO/decompiled" ]; then
    echo "  NOTE: $REPO/decompiled resolves to the tree just written, so it cannot"
    echo "        be a baseline. Skipping the patch diff rather than comparing a"
    echo "        tree against itself and calling it unchanged."
fi

if [ -n "$BASELINE" ]; then
    set +e
    "$PYTHON" "$REPO/scripts/diff_decompiles.py" \
        --old "$BASELINE" --new "$DECOMPILE_DIR" --repo "$REPO" | tee "$OUT/patch-diff.txt"
    rc=${PIPESTATUS[0]}
    set -e
    [ "$rc" -eq 1 ] && drift=1
    "$PYTHON" "$REPO/scripts/diff_decompiles.py" \
        --old "$BASELINE" --new "$DECOMPILE_DIR" --repo "$REPO" \
        --show-changed --json > "$OUT/patch-diff.json" || true

    # ---------------------------------------------------------------------------
    # Exhaustive human-readable changelog for sharing / review.
    # ---------------------------------------------------------------------------
    OLD_VERSION="$BASELINE"
    OLD_STAMP="${BASELINE}/.source.json"
    if [ -f "$OLD_STAMP" ]; then
        OLD_VERSION="$("$PYTHON" -c \
            'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("informational_version","unknown"))' \
            "$OLD_STAMP" 2>/dev/null || echo "unknown")"
    fi
    {
        echo "Slay the Spire 2 Patch Changes Report"
        echo "Generated: $(date -Iseconds)"
        echo "Old build: ${OLD_VERSION}"
        echo "New build: ${VERSION_TAG}"
        echo ""
        echo "This report lists EVERY change between the two game builds."
        echo "ADDED    = new content in this patch"
        echo "REMOVED  = content deleted in this patch"
        echo "CHANGED VALUES = card stats that moved (cost, damage, block, etc.)"
        echo "CHANGED BEHAVIOUR = code changes that may affect gameplay logic"
        echo "[sim has it]      = the simulator already implements this"
        echo "[sim never had it] = the simulator has never seen this"
        echo ""
        # `|| true` is load-bearing. diff_decompiles.py exits 1 when it finds
        # drift, which is the normal outcome of an update, and this group is the
        # last command in it -- so under `set -e` a successful run of the whole
        # script's REASON FOR EXISTING killed it right here.
        #
        # It did, silently, from the commit that added changes.txt until
        # 2026-08-19. The evidence was sitting in reports/: patch-diff.txt and
        # changes.txt dated today, card-parity.txt, inventory.txt and
        # unnamed-content.txt still dated 31 July. Steps 4 to 6 had not run on a
        # drifting build in three weeks, and step 4 is the card parity check --
        # the one thing here that guards the numbers the policy trains on.
        #
        # The exit code is already captured from the first invocation above;
        # this one exists only to write the file.
        "$PYTHON" "$REPO/scripts/diff_decompiles.py" \
            --old "$BASELINE" --new "$DECOMPILE_DIR" --repo "$REPO" \
            --show-changed || true
    } > "$OUT/changes.txt"
else
    echo "  nothing to compare against -- this decompile becomes the baseline."
    echo "  The next run of this script will show what changed."
fi

# ---------------------------------------------------------------------------
step "4/7  Card values (game vs simulator)"
# ---------------------------------------------------------------------------

set +e
"$PYTHON" "$REPO/scripts/check_card_parity.py" \
    --decompiled "$DECOMPILE_DIR" --show-missing | tee "$OUT/card-parity.txt"
rc=${PIPESTATUS[0]}
set -e
[ "$rc" -eq 1 ] && drift=1
[ "$rc" -eq 2 ] && die "card parity check could not run"

# ---------------------------------------------------------------------------
step "5/7  Content inventory (game vs simulator)"
# ---------------------------------------------------------------------------

"$PYTHON" "$REPO/scripts/parity_reference_audit.py" \
    --decompiled-root "$DECOMPILE_DIR" \
    --code-implementation-references --show-missing | tee "$OUT/inventory.txt"

# ---------------------------------------------------------------------------
step "6/7  Content the simulator has no name for"
# ---------------------------------------------------------------------------

# Report only. Adding enum members is additive and safe, but it edits tracked
# source, so it stays an explicit choice rather than something a refresh does to
# you: run scripts/sync_content.py --write when you want it.
"$PYTHON" "$REPO/scripts/sync_content.py" | tee "$OUT/unnamed-content.txt"

# ---------------------------------------------------------------------------
step "7/7  Monster constants (game vs simulator)"
# ---------------------------------------------------------------------------

set +e
"$PYTHON" "$REPO/scripts/audit_monster_constants.py" | tee "$OUT/monster-constants.txt"
rc=${PIPESTATUS[0]}
set -e
[ "$rc" -eq 1 ] && drift=1
[ "$rc" -eq 2 ] && die "monster constant audit could not run"

# ---------------------------------------------------------------------------
echo
echo "=== Reports written to $OUT ==="
echo
if [ "$drift" -eq 1 ]; then
    cat <<'EOF'
What to do with this, in order:

  1. patch-diff.txt REMOVED -- anything marked [sim has it] must come out of
     sts2_env/ first. Content the game deleted but the simulator still deals is
     worse than content it is merely missing: the policy learns to play cards
     that cannot appear.
  2. patch-diff.txt CHANGED VALUES -- this patch's value changes, already
     narrowed to the exact fields. Apply them by hand. These are trustworthy in
     a way the next item is not: both sides came out of the same decompiler, so
     the extractor's blind spots are identical on both and cancel out.
  3. card-parity.txt MISMATCH -- older accumulated drift. Check the decompiled
     source before believing any of them; a card with conditional or multi-hit
     damage will report a false mismatch because the extractor reads only the
     first DamageVar.
  4. patch-diff.txt ADDED and inventory.txt -- new content. Real work, not a
     mechanical edit: behaviour has to be written, not copied.

  5. monster-constants.txt HP RANGE MISMATCH -- trust these. The game declares
     MinInitialHp and MaxInitialHp outright, so there is no extractor guesswork
     of the kind that makes card-parity.txt need a second look. A damage
     difference in the same file is softer: it compares values rather than move
     names, so a renamed move reads as a difference. Read the .cs either way.

  changes.txt is the exhaustive version of patch-diff.txt -- every single class
  that changed, not just the summary. Good for sharing or for grepping when you
  suspect a specific card or power was touched.
EOF
    exit 1
fi
echo "No drift found."
exit 0
