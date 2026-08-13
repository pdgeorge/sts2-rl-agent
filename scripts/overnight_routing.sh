#!/bin/bash
# Overnight, unattended: finish the routing A/B, then sweep the two parameters
# that actually shape the route. No decisions required from anyone until
# morning.
#
# Stage 1 is already running when this starts -- greedy vs planned, 150 paired
# seeds. This waits it out, records the comparison, then runs four planner
# variants two at a time so all 14 usable cores stay busy.
#
# Every stage writes rows incrementally and takes --resume, so a kill at any
# point loses at most the runs in flight.
set -u
cd /media/Bucket_Drive/development/sts2-rl-agent
PY=.venv/bin/python
OUT=output/overnight_routing.txt

say() { echo "[$(date +%H:%M)] $*" | tee -a "$OUT"; }

# ---- Stage 1: wait for the A/B already in flight -------------------------
say "waiting for the greedy/planned A/B to finish"
while pgrep -f "measure_funnel.py .*--tag greedy" >/dev/null 2>&1 \
   || pgrep -f "measure_funnel.py .*--tag planned" >/dev/null 2>&1; do
    sleep 60
done
say "A/B complete"

$PY scripts/compare_funnels.py --tags greedy,planned \
    --rows-prefix output/funnel_routing >> "$OUT" 2>&1

# ---- Stage 2: sweep what shapes the route --------------------------------
# elite value  -- how hard the route is pulled toward relics
# fatal hp     -- how much projected HP counts as "cannot survive this"
#
# Two at a time: 7 workers each against 16 cores, matching stage 1's measured
# throughput of roughly 3 runs per minute per arm.
run_arm() {
    local tag="$1" elite="$2" fatal="$3"
    STS2_PLAN_ELITE_VALUE="$elite" STS2_PLAN_FATAL_HP="$fatal" \
    $PY scripts/measure_funnel.py --runs 150 --seed 80000 --workers 7 \
        --max-nodes 2000 --tag "$tag" --resume --out output/funnel_routing.txt \
        > "output/arm_${tag}.log" 2>&1
}

say "stage 2: sweeping elite value and fatal-HP threshold"
run_arm elite5_fatal34 5.0 0.34 &
run_arm elite3_fatal25 3.0 0.25 &
wait
say "stage 2a done"

run_arm elite2_fatal34 2.0 0.34 &
run_arm elite3_fatal45 3.0 0.45 &
wait
say "stage 2b done"

$PY scripts/compare_funnels.py \
    --tags greedy,planned,elite5_fatal34,elite3_fatal25,elite2_fatal34,elite3_fatal45 \
    --rows-prefix output/funnel_routing >> "$OUT" 2>&1

say "ALL DONE -- read $OUT"
