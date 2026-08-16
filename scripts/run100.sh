#!/bin/bash
# THE live 100-run session. This is the only measurement that counts.
#
#   ./scripts/run100.sh            -> output/*_run100.jsonl
#   ./scripts/run100.sh nightly    -> output/*_nightly.jsonl
#
# STS2 must already be running with the bridge mod listening on 127.0.0.1:9002.
# Ctrl-C stops early and still prints the summary.
#
# Afterwards, the numbers:
#   .venv/bin/python scripts/summarise_live_runs.py output/live_journal_<tag>.jsonl
#
# WHY EACH FLAG IS HERE
#
#   --runs 100          SCOREBOARD.md: n >= 100 for any claim. A 25-run
#                       session resolves nothing finer than ~20 points, and
#                       the 48% session that came from one is now understood
#                       as noise.
#   --live-search       On by default since 2026-08-14, passed explicitly
#                       because it being OPT-IN is the single largest
#                       measurement error this project has made: only 93 of
#                       508 runs ever used the search, so every pooled number
#                       for a month described the v3 model instead. Never
#                       assume the default; state it.
#   --restart-on-crash 30  SIZED FROM THE CRASH RATE, not picked. The
#                       2026-08-16 session crashed 4 times in 36 runs -- once
#                       per 9 -- so 100 runs needs about 12, and 4 ran out
#                       after 5 sessions and 36 runs. 30 is that with headroom.
#
#                       Every crash is the same one and it is not ours:
#                         [RlMap] Agent chose Unknown
#                         Creating NCombatRoom mode=VisualOnly
#                             encounter=PUNCH_OFF_EVENT_ENCOUNTER
#                         EventRoom.EnterInternal -> PunchOff.AfterEventStarted
#                             -> PunchEachOther -> CreatureCmd.TriggerAnim
#                             -> NCreature.SetAnimationTrigger_Patch1
#                         ERROR: Signal '_internal_spine_objects_invalidated'
#                                is already connected
#                       `_Patch1` is BaseLib's Harmony patch against a game
#                       build it predates (BaseLib.dll Jul 31 09:59, game .pck
#                       Jul 31 19:28, 3.4.0 is the newest published). Cleared
#                       of being our AnimationSpeedPatch on 2026-08-11 by
#                       removing that patch and reproducing anyway. PunchOff
#                       fires PunchEachOther from AfterEventStarted, so there
#                       is no choice the agent could make differently -- a `?`
#                       room that rolls Punch Off kills the game on entry.
#                       Restarting is the only lever we hold.
#                       Needs `steam` on PATH.
#   --journal           Per-room, per-fight, per-card record. The run log says
#                       a run reached floor 11; the journal says what happened
#                       on the way, and every analysis in WEEKEND_DECISIONS.md
#                       is built from it.
#   --console-log       Console-only output is how the LiveSearch tracebacks
#                       were lost the first time.
#
# NOT passed, deliberately:
#   --seed              Only for reproducing one run or pairing an A/B. A
#                       fixed seed makes 100 runs measure one map.
#   --stochastic        Deterministic is what we measure; sampling adds
#                       variance to a number already at +/-8 at n=100.
set -u
cd /media/Bucket_Drive/development/sts2-rl-agent

TAG="${1:-run100}"
PY=.venv/bin/python
MODEL=output/combat_v3_overnight/final_model.zip

# A NEW TAG EACH SESSION. These files are appended to, and mixing a new
# session into an old file is how a pooled rate quietly becomes two different
# agents averaged together.
for f in output/live_eval_${TAG}.jsonl output/live_journal_${TAG}.jsonl; do
    if [ -e "$f" ]; then
        echo "REFUSING: $f already exists. Pass a fresh tag: $0 <tag>" >&2
        exit 1
    fi
done

echo "100 live runs, tag '${TAG}'. Ctrl-C stops and still prints the summary."

exec $PY -m sts2_env.bridge.live_eval \
    --model-path "$MODEL" \
    --runs 100 \
    --live-search \
    --restart-on-crash 30 \
    --log "output/live_eval_${TAG}.jsonl" \
    --journal "output/live_journal_${TAG}.jsonl" \
    --console-log "output/live_console_${TAG}.log" \
    --crash-log "output/crash_${TAG}.json"
