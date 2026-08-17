# ROADMAP — the whole arc, act 1 to act 3

This is the long-term plan. It is wax: it will change as measurements come in,
and §8 says explicitly what would change it.

It exists because `PHASE_TWO.md` covers act 1 only, `docs/GLM_ROADMAP_50P_ACT1.md`
is largely executed and also act 1 only, and the external `PROPOSAL.md` plans the
whole arc but on premises the evidence here contradicts. This keeps that
document's good bones — milestone table, gates, risk register, update pipeline —
and corrects the parts that are wrong for this project.

---

## 1. The milestones

| # | milestone | status | pooled evidence |
|---|---|---|---|
| M1 | beat act 1 boss 50% | **13.6% +/- 1.5** | 514 live runs |
| M2A | reach act 2 boss 50% | not measured live | offline 12% of all runs |
| M2B | beat act 2 boss 50% | not measured | — |
| M3A | reach act 3 boss 10% | ~1% | 1 live run in ~120 has reached floor 33 |
| M3B | beat act 3 boss 10% | 0% | — |

These are deliberately not gameable in isolation: M2A punishes an act 1 fix that
strips the deck, M3A punishes an act 2 fix that burns resources. That is the
point of having them.

---

## 2. Architecture, and why it is not up for revision

**Simulator + turn search.** Both halves are load-bearing and both are backed by
measurement rather than preference.

**Search is the agent.** `MODELS.md`: boss win 6.7% -> ~20% when search landed.
Against that, three independent full-run RL attempts plateaued at mean floor
9.5-9.7 (`run_ppo_v4`, 20.5M steps, recorded as "Learned nothing"), and every one
is now unloadable because the observation layout moved. Search is also cheap:
measured 0.08s on a boss turn and 0.52s in the widest position act 1 offers,
against a 3s budget — roughly 35x headroom.

**The simulator is what makes search possible**, and separately it is what makes
iteration affordable: 400 full runs in ~3 h offline against 22 h for the same
number live. Every rejected alternative fails on one of those two properties.

### What this costs, stated honestly

A second model of the game diverges from it. That has been the dominant bug
class all project: 68 unplayable cards, a room-type enum gap, nine wrong monster
HP values, thirteen wrong damages, half the act 1 bosses unmodelled. §5 is the
standing defence against it, and it is permanent overhead, not a phase.

---

## 3. The engine: what actually moves the number

Four wins, all of them **removing something the agent was structurally unable to
do**:

| fix | what was impossible |
|---|---|
| card id resolution | 68 of 600 cards unplayable |
| card reward skip | `can_skip:false` always |
| `?` room RoomType | KeyError -> fight handed to the weak model |
| act 1 variant | 57% of real boss fights unmodelled |

Five nulls, all of them **tuning numbers that were already roughly right**: eval
weights, archetype picking, siphon + potions, incoming damage, search budget.

**So the primary activity at every milestone is hunting impossibilities, not
tuning.** Tuning is what you do once nothing is impossible.

Sources that have each produced a real bug, and should be checked at every
milestone: the disparity log, stuck-state dumps, search-failure captures, the
identifier audit, and a human watching a run and asking why she did not play
that.

---

## 4. Per-milestone plan

### M1 — act 1 at 50%

Covered in full by `PHASE_TWO.md`. Three tracks: explain the 45-point
offline/live boss gap, lift reach from 54% to 75%, decide deck quality.

### M2A — reach act 2 boss 50%

Mostly the same skill as act 1 reach, one act later, so the M1 routing work
should carry. The new content is the risk.

**Known blocker: none.** Act 2 (Hive) matches the game — TheInsatiable,
KnowledgeDemon, KaiserCrab, verified against `Hive.BossDiscoveryOrder`.

**Expected work:** run the identifier audit and disparity log over act 2 content,
which has had almost none of the parity attention act 1 received. The week's
pattern says unaudited content hides exactly this class of bug, and act 2
monsters have already produced disparity reports (Rocket, Crusher, Infested
Prism) that turned out to be correct arithmetic on buffed enemies — which is
evidence the reporting works there, not that the content is clean.

### M2B — beat act 2 boss 50%

Act 2 bosses are larger and scale harder, so deck quality matters more than in
act 1, and the M1 quality-bar result transfers directly.

**Expected work:** whatever M1's Track C concludes, applied with act 2 deck sizes
in view. If M1 concludes that removal and upgrades matter more than declining
cards, this is where that gets built — the deferred "remove the worst card for
this deck" work, which is cheap now that `DeckDirection` is measured to steer
21% of picks.

### M3A / M3B — act 3

**Known blocker, must be fixed before any act 3 number is trusted:** the
simulator's act 3 pool rolls `setup_doormaker_boss`, and Doormaker exists nowhere
in the decompile. The game's third act 3 boss is `AeonglassBoss`, and the
simulator has neither the monster nor the encounter. That is the act 1 variant
bug again — a boss it cannot roll, and one it rolls that does not exist — and it
would silently invalidate a third of act 3 boss measurements.

pd's read is that this is the game having updated (Aeonglass added, Doormaker
removed) rather than fabrication, which makes it catch-up work rather than a
mystery. Either way it is blocking.

**Expected work beyond that:** act 3 is where long-horizon decisions bind — is
this build worth pushing, when to take risk, which archetypes close a run. Those
are the decisions the current agent makes least well, because the search plans
one turn at a time and the run-level policy is a set of heuristics.

---

## 5. Standing defences (permanent, not a phase)

These exist because each one was learned by losing weeks to its absence.

| defence | what it prevents | status |
|---|---|---|
| identifier audit in CI | names the game sends that our model cannot resolve | script exists, not automated |
| derive constants at construction | hand-copied values drifting from the game | cards done; monsters/relics not |
| one function per decision | offline measuring an agent that does not ship | done, test-pinned |
| capture the state that caused a failure | silent degradation with no reproduction | done for search; extend |
| disparity reporting | the simulator quietly disagreeing with the game | done |
| policy version on every run | not knowing which code produced which result | **built 2026-08-16** (live path: journal run_start, eval log and crash log carry `policy_version` + git sha); offline funnel still unstamped |
| weights in versioned config | sweeps monkey-patching globals | **built 2026-08-16**: `policies/v001.json` + `PolicyConfig`; `run_agent --policy` loads it, `apply_active_policy` is the only sanctioned writer of the legacy constants; sweeps still to migrate |
| per-option score logging | not seeing which options never appeared | **built 2026-08-16 for card rewards** (`card_reward_options` journal record, same `rank_cards` path as the decision); other screens follow when a question needs them |
| holdout seeds | tuning and evaluating on the same data | **built 2026-08-16**: `sts2_env/evaluation/seed_split.py` (`seed % 4 == 3`), `compare_funnels.py` reports both halves and calls tuning-only gains NOISE; PHASE_TWO's note stands: the holdout needs more seeds to resolve small effects |

**An identifier audit is not version diffing.** Flame Barrier was never a version
change: the game always sent `FLAME_BARRIER`, we always spelled it
`FLAME_BARRIER_CARD`. A diff between game builds correctly reports "no change"
forever while 68 cards stay unplayable.

---

## 6. Measurement, per milestone

Run cost: **offline ~27 s/run** (400 in ~3 h, 14 workers), **live 2.2 min/run**.

To resolve a target to +/- half its value at 95% confidence:

| target | +/- | runs | offline | live |
|---|---|---|---|---|
| 50% | 5.0 pts | 385 | 0.4 h | 14 h |
| 50% | 2.5 pts | 1537 | 1.7 h | 56 h |
| 10% | 5.0 pts | 139 | 0.2 h | 5 h |
| 10% | 2.5 pts | 554 | 0.6 h | 20 h |

Two consequences:

1. **Live cannot be the measuring device.** A 40-run session resolves ~25 points.
   Live is a bug detector and a sanity check; the offline funnel is the
   instrument. This is the correction to the external proposal, which assumes
   600-2000 "offline" runs through Autoslay — Autoslay is the game, so its Phase
   0 gate alone is 22 hours and its M3 gate is 75.
2. **Rare events are cheaper offline than common ones are live.** M3A at 10%
   needs 139 runs for +/-5 — twelve minutes offline. The act 3 milestones are
   measurable long before they are achievable, which is the right way round.

**Offline is not yet trusted for boss questions.** Reach agrees with live within
ten points; boss win is 45 apart. Until M1 Track A explains that, offline may be
asked reach questions and not boss questions. This constraint applies at every
milestone, not just M1.

---

## 7. Risk register

| risk | severity | evidence | mitigation |
|---|---|---|---|
| offline does not predict live | **high** | boss win 74% vs 29% | M1 Track A blocks on it; restrict offline's remit until resolved |
| act 2/3 parity debt | high | act 3 rolls a boss that does not exist | audit before measuring; treat any act 3 number before that as void |
| game updates break the model | high | BaseLib 3.4.0 broke on a same-day patch; 149 stale test expectations | derive at construction; audit in CI |
| a crash ends a session | medium | Punch Off, ~4 per 100 runs, not ours to fix | `--restart-on-crash`; upstream report filed |
| tuning consumes effort for nothing | medium | five consecutive nulls | hunt impossibilities first; require a mechanism before a sweep |
| small-n conclusions | medium | 25.8% then 14.7% then 0% on consecutive sessions | pooled numbers with n and error bars, always |
| human-likeness vs win rate | low now, high later | not yet measured | quantify the cost of pacing/noise before it is required |

That last one is worth flagging early: the external proposal requires the bot to
"not instantly choose mathematically perfect actions". Deliberately playing
sub-optimally costs win rate. At 14% those coexist comfortably; at 50% they
compete, and at act 3 they compete hard. Someone should measure the cost of the
streaming constraint before it becomes load-bearing.

---

## 8. What would change this plan

Stated in advance so that changing it is a decision rather than a drift.

- **If M1 Track A shows the search wins live's own boss fights**, the fault is in
  the live path and the next month is bridge reconstruction, not agent quality.
- **If it shows the search loses them**, the boss model is honest and offline's
  optimism is upstream in arrival condition — the work moves to HP economy and
  deck strength.
- **If the quality bar makes act 1 worse**, deck size was never the problem and
  M2B's removal/upgrade work is promoted to M1.
- **If act 2 content turns out to be as unaudited as act 1 was**, M2A becomes a
  parity milestone rather than a routing one, and the estimate doubles.
- **If a game update invalidates the decompile**, everything pauses for the
  update pipeline. This has already happened once mid-project.

---

## 9. What this plan will not do

- train a full-run policy (three attempts, floor 9.5-9.7, all now unloadable)
- use weight tuning as the primary engine (five consecutive nulls)
- accept a conclusion from a 30-run live session
- report a favourable subset as the rate
- measure act 3 before act 3's content is audited
