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
#   --restart-on-crash 4  pd's call, 2026-08-16, and it stays 4. A restart
#                       only ever happens AFTER the game has crashed and the
#                       run in flight is already lost, so raising it does not
#                       save runs -- it just keeps relaunching. The crash that
#                       triggers it is the known Punch Off one; see
#                       docs/KNOWN_ISSUES.md, do NOT re-diagnose it, and do not
#                       spend time fixing it before the clear rate hits 50%.
#                       Needs `steam` on PATH.
#   --telemetry         Milestone events (run_start, boss_beaten, elite_beaten,
#                       died, run_end, potion_used, policy_version_loaded) to
#                       the sts2.events topic exchange on RabbitMQ. Fire-and-
#                       forget on a background thread: a dead broker costs one
#                       log line and plays on, never a measurement. See
#                       sts2_env/bridge/telemetry.py. Broker at
#                       STS2_TELEMETRY_URL, default amqp://cyra:changeme@
#                       127.0.0.1:5672/.
#   --journal           Per-room, per-fight, per-card record. The run log says
#                       a run reached floor 11; the journal says what happened
#                       on the way, and every analysis in WEEKEND_DECISIONS.md
#                       is built from it.
#   --console-log       Console-only output is how the LiveSearch tracebacks
#                       were lost the first time.
#   --capture-raw       The boss positions, verbatim, so they can be replayed
#                       offline by scripts/replay_live_boss_fights.py. This is
#                       the ONLY instrument for PHASE_TWO.md's Track A gate --
#                       offline wins ~74% of act 1 bosses and live wins ~31%,
#                       and ~26 of those 45 points are unexplained after HP and
#                       relics are controlled for. It is the largest open number
#                       in the project and 50% is 30 points away.
#                       It was absent until 2026-08-16, so run100 and run100_2
#                       threw away ~75 boss fights between them. Two earlier
#                       bugs kept the capture useless and BOTH are already fixed
#                       in raw_capture.py: the quota is now keyed per fight
#                       (`combat_action:Boss:<encounter_seed>`), not per message
#                       type -- which used to fill on 25 floor-1 monster turns
#                       and keep zero bosses across eleven sessions.
#                       Costs nothing: the session runs regardless.
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
# RawCapture opens its file with "w", so it truncates. Tagged like everything
# else here, or the next session silently erases the boss fights of the last.
for f in output/live_eval_${TAG}.jsonl output/live_journal_${TAG}.jsonl \
         output/bridge_boss_fights_${TAG}.jsonl; do
    if [ -e "$f" ]; then
        echo "REFUSING: $f already exists. Pass a fresh tag: $0 <tag>" >&2
        exit 1
    fi
done

echo "100 live runs, tag '${TAG}'. Ctrl-C stops and still prints the summary."

# NOT `exec` any more. The session is no longer the last thing this script
# does -- the transcripts are rendered afterwards -- and dropping exec also
# means a Ctrl-C still gets them: live_eval catches the interrupt, prints its
# summary and exits normally, and the shell carries on to the export.
$PY -m sts2_env.bridge.live_eval \
    --model-path "$MODEL" \
    --runs 100 \
    --live-search \
    --restart-on-crash 4 \
    --log "output/live_eval_${TAG}.jsonl" \
    --journal "output/live_journal_${TAG}.jsonl" \
    --console-log "output/live_console_${TAG}.log" \
    --crash-log "output/crash_${TAG}.json" \
    --capture-raw "output/bridge_boss_fights_${TAG}.jsonl" \
    --telemetry

# Per-run transcripts, for a reviewing model to read end to end. Seconds to
# render, and doing it here means they are on disk before anyone asks -- the
# alternative is remembering a post-processing step at 3am after a 5-hour run.
#
# Non-fatal on purpose: a rendering bug must never be the reason a finished
# session looks like a failed one. The journal is the artefact that matters and
# it is already written and flushed.
echo
echo "Rendering per-run transcripts..."

# TWO SETS, and the default is act 1 only.
#
# The reviewing model reads output/transcripts/<tag>/, and act 1 is the
# milestone -- a reviewer spending its attention on act 3 corridors is spending
# it on a question we have parked. It is also what keeps a request inside a
# local model's context: the deepest wednesday run rendered to 103 KB, about
# 46k tokens once the game reference is prepended, and llama-server refused it
# at a 40960 window. Cut at the act 1 boss the same run is 39 KB.
$PY scripts/export_run_transcripts.py --tag "${TAG}" --max-floor 17 || \
    echo "WARNING: act 1 transcript export failed; the journal is still intact." >&2

# And the whole run, for reading by hand when something interesting happens
# deeper -- the act 3 Aeonglass run on 2026-08-19 was reconstructed from these.
$PY scripts/export_run_transcripts.py --tag "${TAG}" \
    --out "output/transcripts/${TAG}_full" >/dev/null || \
    echo "WARNING: full transcript export failed." >&2

echo
echo "Numbers:  $PY scripts/summarise_live_runs.py output/live_journal_${TAG}.jsonl"
echo "Review:   output/transcripts/${TAG}/       (act 1 only -- what the model reads)"
echo "By hand:  output/transcripts/${TAG}_full/  (whole runs, all acts)"
echo "Then:     $PY scripts/aggregate_run_reports.py --tag ${TAG}"
