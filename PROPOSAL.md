# PROPOSAL.md

## Project Title

**Offline-Optimized Utility AI for Slay the Spire 2: A Maintainable, Streamable, Fast-Heuristic Auto-Player**

---

## 1. Executive Summary

This proposal describes a research and engineering project to build a Slay the Spire 2 auto-player capable of reaching the following milestones:

1. Beat the Act 1 boss **50%** of the time from full random starts.
2. Reach the Act 2 boss **50%** of the time.
3. Beat the Act 2 boss **50%** of the time.
4. Reach the final boss **10%** of the time.
5. Beat the final boss **10%** of the time.

The proposed system is **not** a live LLM player. It is not a giant end-to-end neural network. It is a hybrid system:

    Decompiled game data
            ↓
    Fast state adapter
            ↓
    Feature extraction
            ↓
    Utility-scored policies
            ↓
    Autoslay offline evaluation
            ↓
    Structured log aggregation
            ↓
    Qwen3.8 Max offline analysis / patch proposal
            ↓
    Policy version update
            ↓
    Offline regression + live sanity testing

The core idea is to build a bot that:

- plays quickly enough for streaming,
- behaves plausibly like a human player,
- is easy to update when STS2 changes,
- improves through measurable offline iteration,
- uses Qwen3.8 Max as an offline researcher/engineer, not as a live turn-by-turn oracle.

This fits your stated constraints:

- no remote LLM during live play,
- full decompiled code available,
- Autoslay can run overnight,
- preference for fast heuristics,
- RabbitMQ telemetry required,
- maintainability is the highest priority.

---

## 2. Problem Definition

We want to build an AI agent that can play Slay the Spire 2 from full random starts and achieve measurable win-rate milestones.

The agent must satisfy several constraints simultaneously:

### 2.1 Performance constraints

- Decisions must be fast.
- No multi-minute turns.
- Target visible decision time usually under 3–5 seconds.
- Complex boss turns may take longer, but rarely.

### 2.2 Behavioral constraints

- The bot must appear plausibly human.
- It should not instantly choose mathematically perfect actions.
- It should show brief deliberation.
- It should occasionally choose a near-optimal but not perfect option.
- It should not visibly stall.

### 2.3 Maintainability constraints

- STS2 is being updated.
- The system must survive patches.
- Updates may trigger retuning or retraining.
- Updates should not destroy the whole project.

### 2.4 Evaluation constraints

- Progress must be measured from full random starts.
- Offline evaluation is preferred.
- Live testing is used for sanity, streaming realism, and final confirmation.
- RabbitMQ events must be emitted at key milestones.

---

## 3. Core Hypothesis

The central research hypothesis is:

> **A parameterized utility-based policy, tuned through repeated offline log analysis and supported by exact game-state features from decompiled code, can reach meaningful STS2 win-rate milestones more reliably than a hand-tuned monolithic script or a live LLM agent.**

More specifically:

### H1: Measurement beats intuition

If we can log every decision and aggregate failures, we can identify the true bottlenecks instead of guessing.

### H2: Fast heuristics can be optimized

If policies are represented as weighted features rather than brittle code paths, we can tune them systematically.

### H3: Offline iteration is the main accelerator

If Autoslay can run hundreds of games overnight, we can use statistical feedback loops to improve the bot.

### H4: Qwen3.8 Max is best used offline

Qwen3.8 Max should analyze logs, propose patches, generate features, and help debug. It should not make every live decision.

### H5: Maintainability requires a stable abstraction layer

The bot should depend on normalized game features, not directly on volatile decompiled names or raw internal structures.

---

## 4. Non-Goals

This project explicitly does **not** aim to:

- use a hosted LLM during live gameplay,
- achieve leaderboard-level hidden optimization,
- create a perfect solver,
- rely on fragile pixel automation if structured state is available,
- train one giant black-box model as the first approach,
- overfit to one STS2 build,
- produce instantaneous robotic play,
- require constant human intervention after the initial framework is built.

---

## 5. Constraints and Assumptions

## 5.1 Constraints

1. **Live remote LLMs are forbidden.**
2. **Qwen3.8 Max is available offline for engineering and analysis.**
3. **A local Qwen 0.6B embedding model exists but should not become a live bottleneck.**
4. **Streaming requires human-like pacing.**
5. **RabbitMQ events must be sent to another service.**
6. **Game updates are frequent.**
7. **The decompiled code is fully usable.**
8. **Autoslay can run games offline for long periods.**

## 5.2 Assumptions

These should be verified during Phase 0:

1. Autoslay can run full runs, not only isolated combats.
2. We can log decisions in a structured format.
3. We can control or at least record game version and seed.
4. We can compare policy versions on comparable seed sets.
5. The bot can execute actions reliably through the chosen interface.
6. Live testing can be done safely and does not violate terms of use.
7. The project will be used offline from leaderboards.

---

## 6. Proposed Approach: Offline-Optimized Utility AI

The proposed architecture is a hybrid of:

- **utility-based AI**,
- **feature engineering from decompiled code**,
- **offline log mining**,
- **local lightweight models**,
- **limited lookahead**,
- **LLM-assisted engineering**,
- **versioned regression testing**.

I would call this approach:

> **Offline-Optimized Utility AI**, or **OOUA**.

It is designed to be:

- fast,
- interpretable,
- tunable,
- patch-friendly,
- streamable,
- statistically measurable.

---

## 7. Why Not Other Approaches?

## 7.1 Why not a live LLM player?

A live LLM player has major problems:

- high latency,
- high token cost,
- unstable reasoning,
- poor maintainability,
- difficulty appearing human-like,
- dependence on network availability,
- hard-to-debug mistakes.

Research in large-scale game agents shows that strong performance usually comes from specialized architectures, simulation, and learned policies rather than general-purpose language reasoning alone. Large language models are useful as assistants, but they are poor candidates for real-time low-level control in this setting.

## 7.2 Why not pure deep reinforcement learning?

Pure RL can be powerful, but it has drawbacks here:

- sample hungry,
- difficult to debug,
- brittle under game patches,
- hard to keep human-like pacing,
- often requires large-scale simulation,
- can learn strange exploitative behavior.

RL is still relevant as a future option, especially for combat value estimation, but I would not start with a full end-to-end RL policy.

Research in deep RL and game playing demonstrates that RL works best when there is a stable simulator, clear rewards, and large compute budgets [1][8][9][10]. STS2 beta churn makes stability harder.

## 7.3 Why not Monte Carlo Tree Search?

MCTS is a strong method for sequential decision-making and has been used in games such as Go and other perfect/imperfect information domains [2][3][4]. AlphaGo-style systems combine search with learned value/policy networks [5][6], and MuZero extends planning with learned models [7].

However, full MCTS is not ideal as the primary live policy here because:

- turns must be fast,
- human-like pacing is required,
- the game state space is large,
- search may produce robotic play,
- STS2 updates may invalidate search assumptions.

A limited lookahead module can still be useful for dangerous combat situations.

## 7.4 Why not only hand-written rules?

Your current 15% method likely uses hand-written rules. Hand-written rules can work, but they become hard to maintain because:

- rules interact unpredictably,
- tuning is manual,
- regressions are hard to detect,
- updates force repeated rewrites,
- it is difficult to know which rule caused a loss.

This is consistent with broader ML/software engineering research showing that complex adaptive systems accumulate hidden technical debt unless carefully structured [15][16].

---

# 8. Proposed Architecture

## 8.1 High-Level Diagram

    +------------------+
    | STS2 / Autoslay  |
    +--------+---------+
             |
             v
    +------------------+
    | Game Adapter     |
    +--------+---------+
             |
             v
    +------------------+
    | State Normalizer |
    +--------+---------+
             |
             v
    +------------------+
    | Feature Layer    |
    +--------+---------+
             |
             v
    +------------------+
    | Utility Policies |
    +--------+---------+
             |
             v
    +------------------+
    | Action Executor  |
    +--------+---------+

    +------------------+       +------------------+
    | Logging System   | ----> | Local Aggregator |
    +------------------+       +--------+---------+
                                        |
                                        v
                               +------------------+
                               | Qwen3.8 Max      |
                               | offline analysis |
                               +------------------+

    +------------------+
    | RabbitMQ Events  |
    +------------------+

## 8.2 Module Responsibilities

### Game Adapter

Talks to STS2/Autoslay.

Responsibilities:

- read current game state,
- send actions,
- detect turn boundaries,
- detect run start/end,
- expose game version.

### State Normalizer

Converts raw decompiled/internal state into stable JSON-like structures.

Responsibilities:

- normalize card names,
- normalize enemy IDs,
- normalize effects,
- handle version differences,
- provide stable feature identifiers.

### Feature Layer

Converts normalized state into numerical and categorical features.

Examples:

- enemy threat,
- block need,
- lethal available,
- deck synergy,
- archetype fit,
- map risk,
- expected gold,
- expected damage,
- rest value,
- boss readiness.

### Utility Policies

Score possible actions using weighted features.

Examples:

    map_score(node)
    card_score(card)
    combat_score(action_sequence)
    shop_score(item)
    rest_score(option)
    relic_score(relic)

### Action Executor

Executes chosen action with human-like pacing.

Responsibilities:

- apply artificial delay,
- optionally hover/reconsider,
- avoid instant perfect clicks,
- log final action.

### Logging System

Writes structured logs for every decision and outcome.

### Local Aggregator

Processes logs into summaries:

- win rates,
- death clusters,
- card pick stats,
- combat mistake stats,
- map choice stats,
- archetype performance,
- regression diffs.

### Qwen3.8 Max Offline Analyzer

Consumes aggregated reports and proposes:

- weight changes,
- new features,
- bug hypotheses,
- code patches,
- tests,
- next experiments.

### RabbitMQ Event Publisher

Emits high-level events:

    run_start
    run_end
    died
    elite_beaten
    boss_beaten
    act_entered
    archetype_chosen
    policy_version_loaded
    update_detected

---

# 9. Research Backing for the Main Design Choices

The following table summarizes why each major design choice is reasonable.

| Design choice | Research/engineering basis | Why it matters here |
|---|---|---|
| Utility-based decision scoring | Utility theory and game AI systems use weighted features to select actions in complex but interpretable ways [18][19]. | Fast, tunable, explainable, suitable for streaming. |
| Feature engineering from decompiled code | Knowledge-rich features reduce sample complexity and make models easier to debug. | STS2 gives exact card/enemy/relic data; using it directly is superior to guessing. |
| Offline simulation/evaluation | Simulator-based training and evaluation are central to modern game AI successes [5][6][7][8][9][10]. | Autoslay allows hundreds/thousands of runs without live pressure. |
| Offline RL / batch learning | Offline RL methods learn from logged experience without active environment interaction [11][12]. | We can improve from recorded runs without requiring live trial-and-error. |
| Bayesian/evolutionary optimization | Bayesian optimization and evolution strategies are effective for tuning black-box parameters [13][14]. | Utility weights can be tuned automatically against win rate. |
| Observability and MLOps discipline | ML systems require monitoring, versioning, testing, and debt management [15][16]. | STS2 updates require maintainability and regression detection. |
| Controlled experiments | Online/offline controlled experiments are standard for validating changes [17]. | We need to know whether a patch actually improved the bot. |
| Limited lookahead | Search can improve tactical decisions when constrained by time budgets [2][3][4]. | Useful for boss fights and dangerous combat turns without slow full MCTS. |
| LLM as offline engineer | LLMs can accelerate coding, log analysis, and hypothesis generation when used with human/tooling oversight. | Qwen3.8 Max becomes a research assistant, not a live bottleneck. |

---

# 10. Milestones and Acceptance Criteria

## 10.1 Milestone Table

| Milestone | Target | Offline acceptance | Live acceptance | Confidence |
|---|---:|---:|---:|---:|
| M0 | Stable evaluation loop | 600-game batch runs, logs aggregate cleanly | 20 live runs with telemetry | High |
| M1 | Beat Act 1 boss | ≥50% over ≥600 offline runs | 20 live sanity runs | 75–85% |
| M2A | Reach Act 2 boss | ≥50% over ≥600 offline runs | 20 live sanity runs | 70–80% |
| M2B | Beat Act 2 boss | ≥50% over ≥600 offline runs | 20 live sanity runs | 55–70% |
| M3A | Reach final boss | ≥10% over ≥2000 offline runs | 20–50 live sanity runs | 45–65% |
| M3B | Beat final boss | ≥10% over ≥2000 offline runs | 20–50 live sanity runs | 30–50% |

## 10.2 Statistical Notes

For a binomial estimate:

- 600 runs at 50% win rate gives roughly ±4% confidence interval at 95% confidence.
- 600 runs at 15% win rate gives roughly ±3% confidence interval.
- 2000 runs at 10% gives roughly ±1.3% standard error, about ±2.6% at 95% confidence.

Live tests of 20 runs are useful for sanity and realism, but not sufficient to prove a 50% milestone. They should detect catastrophic issues, not replace offline statistical evaluation.

---

# 11. Phase Plan

## Phase 0: Instrumentation, Baseline, and Evaluation Harness

## 11.0.1 Objective

Build the measurement system before trying to make the bot smarter.

The first job is to turn the project into an experiment platform.

## 11.0.2 Tasks

1. Define structured log schema.
2. Add policy version to every run.
3. Add game version to every run.
4. Add run ID, seed if available, character, difficulty.
5. Emit RabbitMQ events.
6. Run 600 offline games with current policy.
7. Build local log parser.
8. Build aggregate report generator.
9. Identify top death clusters.
10. Confirm current Act 1 boss win rate.

## 11.0.3 Deliverables

- LOG_SCHEMA.md
- EVENT_SCHEMA.md
- baseline_report.md
- aggregated metrics dashboard/files
- first 600-run offline dataset
- 20-run live sanity report

## 11.0.4 Acceptance Criteria

The project passes Phase 0 when:

- 600 offline games can run unattended,
- every run produces structured logs,
- logs can be aggregated automatically,
- RabbitMQ events are emitted correctly,
- we can identify the top 10 failure patterns,
- baseline Act 1 boss win rate is measured.

## 11.0.5 Research backing

Instrumentation and observability are foundational. ML systems often fail due to hidden technical debt, poor monitoring, and weak deployment discipline [15][16]. Controlled experiments require reliable metrics [17].

In game AI, strong agents are usually built on top of reproducible environments and evaluation pipelines [5][6][7][8][9][10].

## 11.0.6 How I would tackle it

I would start by defining a compact but complete JSON Lines log format.

Example run record:

    {
      "run_id": "run_2026_08_12_0001",
      "policy_version": "v0.0.0",
      "game_version": "0.x.y",
      "seed": "unknown",
      "character": "unknown",
      "start_time": "2026-08-12T00:00:00Z",
      "outcome": "died_act1",
      "floor_reached": 6,
      "act": 1,
      "death_cause": "elite",
      "boss_killed": null
    }

Example decision record:

    {
      "run_id": "run_2026_08_12_0001",
      "decision_id": 142,
      "type": "card_reward",
      "timestamp": "2026-08-12T00:12:31Z",
      "state_features": {
        "hp": 58,
        "max_hp": 75,
        "deck_size": 13,
        "gold": 66,
        "act": 1,
        "floor": 4
      },
      "options": [
        {
          "card_id": "card_a",
          "score": 0.72,
          "features": {
            "damage_value": 0.61,
            "synergy_value": 0.80,
            "curve_fit": 0.55
          }
        },
        {
          "card_id": "card_b",
          "score": 0.44,
          "features": {
            "damage_value": 0.30,
            "synergy_value": 0.51,
            "curve_fit": 0.48
          }
        },
        {
          "card_id": "skip",
          "score": 0.51
        }
      ],
      "chosen": "card_a",
      "latency_ms": 143
    }

The local aggregator would then produce reports such as:

    Total runs: 600
    Act 1 boss reached: 35.2%
    Act 1 boss won: 15.1%

    Top death clusters:
    1. Died before boss to elite with HP < 35%.
    2. Died to Act 1 boss on turn 3 due to underblocking.
    3. Took too many low-synergy cards before boss.
    4. Unused healing potion at death.
    5. Poor target priority in multi-enemy fights.

That report is what Qwen3.8 Max should analyze.

---

## Phase 1: Convert Policies into Parameterized Utility Functions

## 11.1.1 Objective

Replace opaque hardcoded logic with explicit weighted scoring functions.

## 11.1.2 Tasks

1. Inventory current decision rules.
2. Identify major decision types.
3. Convert rules into features.
4. Store weights in config files.
5. Make policy versionable.
6. Add fallback behavior for missing features.
7. Create unit tests for feature extraction.

## 11.1.3 Deliverables

- feature catalog,
- weight config files,
- policy versioning system,
- unit tests,
- refactored decision modules.

## 11.1.4 Acceptance Criteria

- The bot can run with policy_v001.
- Changing weights does not require rewriting code.
- Each major decision logs all option scores.
- Feature extraction is deterministic for the same state.

## 11.1.5 Research backing

Utility systems are common in game AI because they are fast, interpretable, and tunable [18][19]. Parameterized policies are also easier to optimize with Bayesian or evolutionary methods [13][14].

This reduces hidden technical debt compared with deeply nested hardcoded rules [15][16].

## 11.1.6 How I would tackle it

I would define separate policy modules:

    policy_map
    policy_card_reward
    policy_shop
    policy_rest
    policy_combat
    policy_relic
    policy_event

Each module returns:

    {
      "decision_type": "card_reward",
      "scores": [...],
      "chosen": "...",
      "debug": {...}
    }

The weights file might look like:

    {
      "card_reward": {
        "damage_value": 1.00,
        "block_value": 0.90,
        "scaling_value": 1.20,
        "synergy_value": 1.50,
        "curve_fit": 0.80,
        "skip_threshold": 0.45
      },
      "combat": {
        "block_need": 1.30,
        "lethal_priority": 1.70,
        "aoe_value": 0.90,
        "potion_hoarding": 0.40
      },
      "map": {
        "elite_risk": 1.20,
        "rest_value": 1.00,
        "merchant_value": 0.70,
        "treasure_value": 0.90
      }
    }

The current 15% method would not be discarded immediately. It would be converted into an initial policy configuration.

---

## Phase 2: Build the Offline Analysis Engine

## 11.2.1 Objective

Turn raw logs into actionable reports.

## 11.2.2 Tasks

1. Parse JSON logs.
2. Store in SQLite/Parquet/CSV.
3. Compute win rates by decision, card, relic, map path, archetype.
4. Detect death clusters.
5. Detect wasted resources.
6. Detect missed lethal.
7. Detect underblocking.
8. Detect bad card picks.
9. Detect risky map choices.
10. Produce markdown reports.

## 11.2.3 Deliverables

- log parser,
- metrics database,
- failure clustering report,
- card/relic performance report,
- combat mistake report,
- map mistake report.

## 11.2.4 Acceptance Criteria

Given a batch of 600 runs, the system can automatically answer:

- Why did runs die?
- Which cards correlate with losses?
- Which relics correlate with wins?
- Where is the bot underblocking?
- Where is it taking unnecessary elites?
- Which bosses are problematic?
- Which archetypes perform best?

## 11.2.5 Research backing

Offline learning and log analysis are central to batch/offline RL, where policies improve from stored datasets rather than live interaction [11][12]. Even without full RL, structured log mining allows supervised and heuristic optimization.

Observability is also a known requirement for maintainable ML systems [15][16].

## 11.2.6 How I would tackle it

I would not send raw logs to Qwen3.8 Max.

Instead:

    raw logs
      -> local parser
      -> structured tables
      -> aggregate statistics
      -> sampled failure cases
      -> Qwen3.8 Max report prompt

The prompt to Qwen3.8 Max would contain:

- current weights,
- current feature definitions,
- top failure clusters,
- metric deltas from previous version,
- 10–30 sampled bad decisions,
- recent code changes,
- milestone goals.

This keeps token cost controlled and analysis focused.

---

## Phase 3: Create the Optimization Loop

## 11.3.1 Objective

Create a repeatable process for improving the bot.

## 11.3.2 Proposed loop

    1. Run 600 offline games with policy vN.
    2. Aggregate logs.
    3. Compare against previous policy.
    4. Qwen3.8 Max proposes patch.
    5. Apply patch as policy vN+1.
    6. Run same/rotated seed set.
    7. Accept only if target metrics improve.
    8. Run 20 live sanity games.
    9. Repeat.

## 11.3.3 Deliverables

- optimization playbook,
- policy comparison script,
- patch proposal format,
- regression acceptance rules.

## 11.3.4 Acceptance Criteria

The loop is successful when:

- one full offline cycle can run overnight,
- Qwen3.8 Max can propose actionable patches,
- we can compare policy versions automatically,
- bad patches are rejected before live testing.

## 11.3.5 Research backing

Bayesian optimization and evolution strategies are useful for tuning parameters when the objective is noisy but measurable [13][14]. Controlled experiments help avoid false positives [17].

Using an LLM as an offline assistant is a practical engineering accelerator, but the loop must include automated validation.

## 11.3.6 How I would tackle it

At first, optimization would be semi-manual:

    Qwen proposes weight changes.
    Human/automation applies them.
    Offline evaluator validates them.

Later, we can automate more:

    Qwen proposes JSON patch
      -> optimizer applies patch
      -> offline evaluator runs
      -> comparison script accepts/rejects

Example patch format:

    {
      "policy_version": "v0.1.4",
      "based_on": "v0.1.3",
      "weight_changes": {
        "combat.block_need": "+0.15",
        "card_reward.skip_threshold": "-0.05",
        "map.elite_risk": "+0.10"
      },
      "code_changes": [],
      "expected_effect": "Reduce early boss deaths from underblocking.",
      "metrics_to_watch": [
        "act1_boss_win_rate",
        "avg_hp_at_act1_boss",
        "unused_healing_potion_rate",
        "pre_boss_death_rate"
      ]
    }

---

## Phase 4: Milestone 1 — Beat Act 1 Boss 50%

## 11.4.1 Objective

Reach a 50% Act 1 boss win rate from full random starts.

## 11.4.2 Likely bottlenecks

Based on common roguelike deckbuilder failure modes:

1. Underblocking high-damage intents.
2. Taking too many low-value cards.
3. Poor early elite risk.
4. Unused potions.
5. Weak target priority.
6. Poor card skip discipline.
7. Lack of boss-specific logic.

## 11.4.3 Tasks

1. Analyze death clusters.
2. Improve combat block threshold.
3. Add enemy intent danger feature.
4. Improve card reward skip threshold.
5. Reduce deck bloat.
6. Add potion usage rules.
7. Add Act 1 boss-specific behavior.
8. Run repeated offline optimization.

## 11.4.4 Deliverables

- improved combat utility policy,
- improved card reward policy,
- improved early map policy,
- Act 1 boss-specific modules,
- offline report showing ≥50% win rate,
- live sanity report.

## 11.4.5 Acceptance criteria

Offline:

    Act 1 boss win rate >= 50%
    over >= 600 runs
    with stable or improved pre-boss survival

Live:

    20 live runs show no catastrophic behavior,
    reasonable pacing,
    no obvious non-human timing.

## 11.4.6 Research backing

Improving from a weak baseline often comes from eliminating obvious tactical mistakes before adding complex strategy. In game AI, tactical competence is often achieved through better state evaluation and search/value estimation [1][5][6][7][8].

Utility scoring with threat features is a practical way to encode tactical knowledge without slow reasoning [18][19].

## 11.4.7 Expected progression

A plausible progression:

    Current method: 15%
    After measurement + obvious fixes: 20–30%
    After weight tuning: 30–40%
    After boss-specific combat logic: 45–55%

This is not guaranteed, but it is a reasonable research trajectory.

---

## Phase 5: Milestone 2A — Reach Act 2 Boss 50%

## 11.5.1 Objective

Reach the Act 2 boss in at least 50% of full random runs.

## 11.5.2 Main skills required

- map routing,
- risk management,
- resource conservation,
- deck quality control,
- rest/shop planning,
- potion economy.

## 11.5.3 Tasks

1. Build map node scoring.
2. Add expected damage features.
3. Add elite risk features.
4. Add rest value features.
5. Add merchant value features.
6. Add HP pressure features.
7. Add deck quality score.
8. Tune map policy offline.

## 11.5.4 Deliverables

- map utility policy,
- route evaluation report,
- elite risk report,
- resource economy report,
- offline Act 2 reach rate ≥50%.

## 11.5.5 Acceptance criteria

    Reach Act 2 boss >= 50%
    over >= 600 runs
    without unacceptable HP/resource collapse

## 11.5.6 Research backing

Map routing is a sequential decision problem. Full search is expensive, but utility-based lookahead can approximate good behavior quickly. This is consistent with utility AI practice in game systems [18][19] and with bounded search methods [2][3][4].

Offline evaluation allows us to estimate expected route outcomes statistically [11][12][17].

---

## Phase 6: Milestone 2B — Beat Act 2 Boss 50%

## 11.6.1 Objective

Beat the Act 2 boss in at least 50% of full random runs.

## 11.6.2 Why this is harder

Act 2 boss performance depends on:

- deck scaling,
- relic quality,
- removal/upgrades,
- archetype coherence,
- combat execution,
- prior resource usage.

## 11.6.3 Tasks

1. Add archetype tracking.
2. Improve card synergy features.
3. Improve relic scoring.
4. Improve upgrade/removal choices.
5. Add Act 2 boss-specific logic.
6. Add limited combat lookahead.
7. Improve potion usage in boss fights.

## 11.6.4 Deliverables

- archetype state tracker,
- improved deck policy,
- boss policy modules,
- combat lookahead module,
- offline Act 2 boss win rate ≥50%.

## 11.6.5 Acceptance criteria

    Beat Act 2 boss >= 50%
    over >= 600 runs
    with stable Act 2 reach rate

## 11.6.6 Research backing

Boss fights often require tactical lookahead. Limited-depth search can improve tactical decisions while staying within time budgets [2][3][4]. Learned or heuristic value functions can guide search and utility scoring [5][6][7].

Archetype tracking is a form of higher-level state abstraction. It helps the bot make coherent long-term choices rather than locally greedy ones.

---

## Phase 7: Milestone 3A — Reach Final Boss 10%

## 11.7.1 Objective

Reach the final boss in at least 10% of full random runs.

## 11.7.2 Main challenge

This milestone requires long-horizon planning.

The bot must understand:

- whether a build can scale,
- whether a run has high potential,
- when to take risks,
- when to conserve resources,
- which archetypes can close out runs.

## 11.7.3 Tasks

1. Build run potential estimator.
2. Improve archetype viability ranking.
3. Improve late-game card scoring.
4. Improve relic synergy scoring.
5. Improve rest/upgrade planning.
6. Improve elite risk based on build strength.
7. Add run-stage-aware policies.

## 11.7.4 Deliverables

- run potential model,
- archetype viability table,
- late-game policy adjustments,
- offline final boss reach rate ≥10%.

## 11.7.5 Acceptance criteria

    Reach final boss >= 10%
    over >= 2000 offline runs
    with stable Act 2 metrics

## 11.7.6 Research backing

Long-horizon decision-making benefits from value estimation and planning [1][5][6][7]. Offline logs can be used to estimate state value or run success probability [11][12].

Because final boss reach is a rare-event metric, larger sample sizes are needed for stable estimation [17].

---

## Phase 8: Milestone 3B — Beat Final Boss 10%

## 11.8.1 Objective

Beat the final boss in at least 10% of full random runs.

## 11.8.2 Main challenge

This is the hardest milestone because it requires:

- strong build ceiling,
- good relic synergy,
- low accumulated damage,
- correct potion usage,
- strong final boss mechanics,
- stable execution over a long run.

## 11.8.3 Tasks

1. Analyze final boss deaths.
2. Build final boss-specific policy.
3. Improve late-game defensive thresholds.
4. Improve burst windows.
5. Improve potion timing.
6. Improve run-closing archetype selection.
7. Run large offline validation.

## 11.8.4 Deliverables

- final boss policy module,
- final boss combat report,
- improved late-game deck policy,
- offline final boss win rate ≥10%.

## 11.8.5 Acceptance criteria

    Beat final boss >= 10%
    over >= 2000 offline runs
    with live sanity confirmation

## 11.8.6 Research backing

High-level game performance often requires a combination of policy evaluation, planning, and domain-specific features [5][6][7][8][9][10]. In a changing beta environment, maintaining a modular policy stack is safer than relying on one opaque model [15][16].

This milestone should be treated as research-grade. It is possible, but not guaranteed on a fixed schedule.

---

# 12. RabbitMQ Event Design

## 12.1 Event requirements

The system must emit high-level events to another service.

Recommended events:

    run_start
    run_end
    died
    elite_beaten
    boss_beaten
    act_entered
    archetype_chosen
    card_taken
    card_skipped
    relic_taken
    potion_used
    shop_purchase
    rest_taken
    policy_version_loaded
    update_detected

## 12.2 Example event payload

    {
      "event_type": "boss_beaten",
      "run_id": "run_2026_08_12_0001",
      "policy_version": "v0.3.1",
      "game_version": "0.x.y",
      "timestamp": "2026-08-12T01:11:32Z",
      "act": 1,
      "boss_id": "boss_act1_x",
      "archetype": "poison_scaling",
      "hp_remaining": 41,
      "floor": 16
    }

## 12.3 Archetype event

The archetype_chosen event is special because it is inferred, not directly emitted by the game.

Example:

    {
      "event_type": "archetype_chosen",
      "run_id": "run_2026_08_12_0001",
      "timestamp": "2026-08-12T00:21:10Z",
      "archetype": "strength_scaling",
      "confidence": 0.74,
      "evidence": [
        "relic_strength_focus",
        "card_strike_synergy_3",
        "card_attack_buff_2"
      ]
    }

The archetype module should only emit this event when confidence is stable enough. Otherwise it will spam downstream services with premature commitments.

## 12.4 Research/engineering rationale

Event-driven telemetry supports monitoring, regression detection, and downstream analysis. It also aligns with MLOps best practices around observability and model/version tracking [15][16].

---

# 13. Logging Schema

A minimal but powerful schema should include:

## 13.1 Run-level fields

    run_id
    policy_version
    game_version
    seed
    character
    difficulty
    start_time
    end_time
    outcome
    act_reached
    floor_reached
    boss_killed
    death_cause
    archetype_final

## 13.2 Decision-level fields

    decision_id
    decision_type
    timestamp
    state_features
    available_options
    option_scores
    chosen_option
    latency_ms

## 13.3 Combat-level fields

    combat_id
    turn_number
    player_hp
    player_block
    energy
    hand_cards
    enemy_hp
    enemy_intent
    enemy_buffs
    enemy_debuffs
    played_cards
    damage_dealt
    damage_taken
    block_gained
    block_wasted
    lethal_available
    lethal_missed

## 13.4 Card reward fields

    reward_id
    offered_cards
    card_scores
    archetype_probabilities
    deck_size
    chosen_card
    skipped

## 13.5 Map fields

    map_choice_id
    available_nodes
    node_scores
    chosen_node
    hp_before
    expected_risk

---

# 14. Feature Engineering Plan

The feature layer is the heart of the project.

## 14.1 Combat features

Examples:

    enemy_intent_damage
    enemy_intent_debuff
    enemy_hp_percent
    lethal_available
    lethal_overflow
    block_needed
    block_efficiency
    energy_available
    card_draw_value
    aoe_value
    single_target_value
    vulnerable_value
    weak_value
    strength_value
    poison_value
    exhaust_value
    time_to_kill
    danger_score

## 14.2 Card reward features

Examples:

    base_power
    upgraded_power
    energy_cost
    damage_per_energy
    block_per_energy
    scaling_value
    synergy_with_deck
    synergy_with_relics
    archetype_fit
    curve_fit
    duplicate_penalty
    deck_size_penalty
    boss_fit
    elite_fit
    opportunity_cost

## 14.3 Map features

Examples:

    node_type
    path_length
    expected_damage
    expected_gold
    expected_card_reward
    elite_risk
    treasure_value
    merchant_value
    rest_value
    event_risk
    hp_pressure
    boss_readiness
    deck_power_score

## 14.4 Run-level features

Examples:

    current_act
    floor
    gold
    hp_percent
    deck_size
    relic_count
    potion_count
    archetype_confidence
    build_power
    scaling_power
    burst_power
    defensive_power
    run_potential

---

# 15. Local Model Plan

The live bot should remain fast. Any model used live must be tiny.

## 15.1 Candidate local models

1. **Card value model**
   - Input: card features + deck state.
   - Output: card usefulness.

2. **Combat danger model**
   - Input: combat state.
   - Output: probability of taking dangerous damage.

3. **Run potential model**
   - Input: run state.
   - Output: probability of reaching next milestone.

4. **Archetype classifier**
   - Input: deck/relics/card history.
   - Output: archetype probabilities.

## 15.2 Model types

Preferred:

- small decision trees,
- gradient-boosted trees,
- tiny MLPs,
- logistic regression,
- cached lookup tables.

Avoid live:

- large transformers,
- remote API calls,
- expensive text inference,
- large embedding computation per decision.

## 15.3 Role of Qwen 0.6B embeddings

Your existing Qwen 0.6B embedding model can be useful, but mostly offline.

Possible uses:

- card similarity clustering,
- archetype suggestion,
- text-effect grouping,
- bootstrapping synergy features.

However, because decompiled code gives exact structured data, structured features should usually dominate over text embeddings.

Recommended use:

    Offline:
      card text/effects -> embeddings -> synergy clusters

    Live:
      cached synergy scores -> fast utility policy

---

# 16. Optimization Methods

## 16.1 Stage 1: LLM-assisted heuristic tuning

Qwen3.8 Max reads reports and proposes changes.

Good for:

- quick patches,
- bug hypotheses,
- feature ideas,
- obvious mistakes.

## 16.2 Stage 2: Manual weight tuning

Use offline metrics to tune weights semi-manually.

Good for:

- early iteration,
- understanding feature impact,
- building intuition.

## 16.3 Stage 3: Automated local search

Use simple search over weights:

    random perturbation
    hill climbing
    simulated annealing
    evolutionary search
    CMA-ES

Good for:

- improving weighted utility policies,
- finding robust configurations,
- reducing manual effort.

Research support: evolutionary strategies can be effective for black-box optimization of policies [14].

## 16.4 Stage 4: Bayesian optimization

Use Bayesian optimization for expensive/noisy objectives [13].

Good for:

- tuning a small number of important weights,
- optimizing milestone metrics,
- balancing exploration and exploitation.

## 16.5 Stage 5: Supervised/offline value models

Train small models to predict:

- win probability,
- combat survival,
- boss success,
- run potential.

Research support: offline RL and batch learning can leverage logged data [11][12].

## 16.6 Stage 6: Limited lookahead

Add shallow search only for high-stakes decisions.

Examples:

- boss fights,
- lethal puzzles,
- dangerous enemy intents,
- low HP situations.

Research support: bounded search improves tactical performance without requiring full MCTS [2][3][4].

---

# 17. Human-Like Streaming Behavior

The bot must not look like an inhuman optimizer.

## 17.1 Timing policy

Recommended visible delays:

| Decision type | Compute budget | Visible delay |
|---|---:|---:|
| Map choice | <200 ms | 0.8–2.5 s |
| Card reward | <200 ms | 1.0–3.0 s |
| Shop choice | <200 ms | 1.0–3.0 s |
| Rest choice | <200 ms | 0.8–2.0 s |
| Normal combat turn | <300 ms | 0.8–2.5 s |
| Complex combat turn | <1 s | 1.5–4.0 s |
| Boss special turn | <2 s | 2.0–5.0 s |

## 17.2 Choice noise

The bot should not always choose the exact top score.

Use:

    If top score - second score < epsilon:
        choose among top 2–3 options
    Else:
        choose top option

Optional:

- small probability of choosing second-best,
- small probability of brief reconsideration,
- variable think delays,
- no instant perfect reactions.

## 17.3 Why this matters

Human-like pacing improves stream verisimilitude and avoids the appearance of tool-assisted instant optimization. It also matches your requirement that the AI play visibly like a person.

---

# 18. Maintainability Across Game Updates

This is the highest priority.

## 18.1 Design rule

The bot must not depend directly on unstable names or structures.

Instead:

    raw decompiled data
      -> generated normalized game data
      -> feature extraction
      -> policy

## 18.2 Update pipeline

When STS2 updates:

1. Detect game version change.
2. Regenerate normalized card/enemy/relic data.
3. Diff old and new data.
4. Run regression suite with old policy.
5. Identify broken mechanics.
6. Update features or rules if needed.
7. Retune weights.
8. Compare against previous metrics.
9. Promote new policy only if stable.

## 18.3 Regression suite

Recommended regression suite:

    100 fixed seeds for smoke test
    300 seeds for quick validation
    600 seeds for milestone validation
    2000 seeds for final boss metrics

## 18.4 Policy fallback

Always keep previous policy versions:

    policy_v010
    policy_v011
    policy_v012

If a patch breaks the new policy, fall back while diagnosing.

## 18.5 Research backing

Maintainability requires versioning, testing, and monitoring. This is well-documented in ML engineering literature [15][16]. Controlled regression testing helps avoid silent performance drops [17].

---

# 19. Experimental Design

## 19.1 Baseline

Baseline policy:

    policy_current_15_percent

Metrics:

    act1_boss_win_rate
    act2_reach_rate
    act2_boss_win_rate
    final_reach_rate
    final_win_rate
    avg_floor_reached
    avg_hp_at_boss
    death_cause_distribution

## 19.2 Paired seed comparison

When possible, compare policies on the same seed set:

    policy_vN   -> 600 seeds
    policy_vN+1 -> same 600 seeds

This reduces variance.

## 19.3 Holdout seeds

Keep a separate holdout seed set to detect overfitting:

    training seeds: 600
    validation seeds: 200
    holdout seeds: 200
    live sanity runs: 20

## 19.4 Statistical acceptance

A change should not be accepted based on tiny improvements.

Suggested thresholds:

- Act 1 boss milestone: require ≥50% over 600 runs.
- Act 2 milestones: require ≥50% over 600 runs.
- Final boss milestones: require ≥10% over 2000 runs.
- Intermediate improvements: require meaningful delta beyond confidence noise.

## 19.5 Research backing

Controlled experiments and statistical validation are essential to avoid false conclusions [17]. Offline evaluation is standard in game AI and RL research [5][6][7][8][9][10].

---

# 20. Timeline Estimate

Assuming active development and stable Autoslay access:

| Phase | Description | Time estimate |
|---|---|---:|
| Phase 0 | Instrumentation and baseline | 1–3 weeks |
| Phase 1 | Parameterized utility policies | 2–4 weeks |
| Phase 2 | Offline analysis engine | 1–3 weeks |
| Phase 3 | Optimization loop | 1–2 weeks |
| Milestone 1 | Act 1 boss 50% | 2–8 weeks |
| Milestone 2A | Reach Act 2 boss 50% | 2–4 weeks |
| Milestone 2B | Beat Act 2 boss 50% | 3–8 weeks |
| Milestone 3A | Reach final boss 10% | 5–10 weeks |
| Milestone 3B | Beat final boss 10% | 6–20+ weeks |

Total realistic timeline:

    4–9 months if progress is strong
    6–12 months including beta update churn

---

# 21. Cost Estimate

This estimate assumes Qwen3.8 Max is used offline for engineering and log analysis, not live gameplay.

## 21.1 Token budgets

| Phase/Milestone | Token budget estimate |
|---|---:|
| Phase 0 | 5M–15M |
| Phase 1 | 10M–30M |
| Phase 2 | 5M–20M |
| Phase 3 | 5M–20M |
| Milestone 1 | 20M–60M |
| Milestone 2A | 10M–35M |
| Milestone 2B | 25M–80M |
| Milestone 3A | 30M–120M |
| Milestone 3B | 60M–300M |

Total:

    170M–680M tokens

## 21.2 Cost planning

Because API pricing varies, use your actual Qwen3.8 Max blended cost.

Example planning ranges:

| Blended cost | Rough total API cost |
|---:|---:|
| $1 / 1M tokens | $170–$680 |
| $3 / 1M tokens | $510–$2,040 |
| $5 / 1M tokens | $850–$3,400 |
| $10 / 1M tokens | $1,700–$6,800 |

Practical planning budget:

    Lean: $500–$2,000
    Serious: $2,000–$8,000
    Heavy agentic loop: $8,000–$20,000+

The most important cost control is:

> Do not feed raw 600-game logs directly into the LLM. Aggregate locally first.

---

# 22. Risk Register

| Risk | Severity | Mitigation |
|---|---:|---|
| Autoslay cannot log enough state | High | Phase 0 must verify logging before major investment. |
| Game updates break adapter frequently | High | Stable normalized schema, regression suite, policy fallback. |
| Offline results do not match live behavior | Medium | Use live sanity runs and latency/behavior checks. |
| Qwen proposes plausible but wrong patches | Medium | Require offline validation before acceptance. |
| Token cost grows from raw log analysis | Medium | Use local aggregation and sampled failures. |
| Bot becomes too robotic | Medium | Add pacing, choice noise, reconsideration behavior. |
| Final boss RNG dominates | Medium/High | Treat final boss beat as research milestone. |
| Overfitting to offline seeds | Medium | Holdout seeds, seed rotation, live sanity. |
| Current method too rigid to refactor | Medium | Wrap old rules as one policy version, do not delete immediately. |

---

# 23. Go/No-Go Gates

## Gate 0: Measurement gate

Proceed if:

- 600 offline games run successfully,
- logs are complete,
- baseline metrics are stable,
- RabbitMQ events work.

Otherwise:

- fix instrumentation before continuing.

## Gate 1: First improvement gate

Proceed if:

- at least one policy change improves a major metric,
- no serious regression appears,
- live behavior remains plausible.

Otherwise:

- re-examine feature design or combat state parsing.

## Gate 2: Act 1 gate

Proceed to Act 2 if:

- offline Act 1 boss win rate ≥50%,
- live sanity runs are stable.

Otherwise:

- continue Act 1 optimization or add limited combat lookahead.

## Gate 3: Act 2 gate

Proceed to final boss research if:

- Act 2 reach ≥50%,
- Act 2 boss win ≥50%,
- archetype tracking is stable.

Otherwise:

- improve archetype planning and boss-specific combat.

## Gate 4: Final boss research gate

Proceed to final boss beat if:

- final boss reach ≥10%,
- run potential model is useful,
- late-game archetypes are viable.

Otherwise:

- treat final boss beat as long-term research.

---

# 24. Detailed First 2-Week Pilot Plan

This is the safest way to begin.

## Week 1

### Goal

Build measurement.

### Tasks

1. Define log schema.
2. Add run metadata.
3. Add decision logs.
4. Add RabbitMQ events.
5. Run 600 offline games.
6. Build local parser.
7. Generate baseline report.

### Deliverable

    baseline_report.md

Should include:

    Current Act 1 boss win rate: 15.x%
    Top death causes:
    ...
    Top bad decisions:
    ...
    Top suspicious cards/relics:
    ...

## Week 2

### Goal

Run first optimization cycle.

### Tasks

1. Convert a few critical rules into weighted features.
2. Send aggregate report to Qwen3.8 Max.
3. Generate first patch proposal.
4. Apply patch.
5. Rerun 600 offline games.
6. Compare with baseline.
7. Run 20 live games.

### Deliverable

    iteration_001_report.md

Should include:

    Policy v0.0.1 changes:
    ...
    Offline metric delta:
    ...
    Live sanity observations:
    ...
    Next hypothesis:
    ...

Success for the pilot is not necessarily reaching 50% immediately.

Success is:

- the loop works,
- metrics move,
- we understand the next bottleneck.

---

# 25. Example Qwen3.8 Max Analysis Prompt

This is the kind of prompt I would use after local aggregation:

    You are optimizing a Slay the Spire 2 bot.
    The bot must use fast local heuristics.
    Do not propose live remote LLM usage.
    Do not rewrite the entire architecture unless absolutely necessary.

    Current milestone: Act 1 boss 50%.
    Current offline win rate: 18.2% over 600 runs.
    Previous win rate: 15.1%.

    Feature list:
    ...

    Current weights:
    ...

    Top death clusters:
    1. Died to Act 1 boss on turn 3 due to underblocking.
    2. Died before boss after taking unnecessary elite.
    3. Died with unused healing potion.
    4. Took low-synergy cards that increased deck bloat.
    5. Missed lethal on turn 5 in 42 cases.

    Sample bad decisions:
    ...

    Propose the smallest high-impact changes.
    Output:
    1. JSON weight changes.
    2. Optional code patches.
    3. Expected effect.
    4. Metrics to watch.
    5. Risks.

This keeps the model focused on controlled engineering changes.

---

# 26. What Success Looks Like

The project is successful if:

- the bot improves measurably,
- each update does not require a rewrite,
- offline runs explain why the bot wins or loses,
- live runs look plausible,
- RabbitMQ telemetry works,
- milestone progress is statistically visible,
- Qwen3.8 Max accelerates development without becoming a live dependency.

---

# 27. Final Recommendation

I would proceed in this order:

1. **Do not start by trying to beat the final boss.**
2. **Do not start with a giant neural network.**
3. **Do not use live remote LLMs.**
4. **Start with measurement.**
5. **Convert behavior into tunable utility policies.**
6. **Use Autoslay to run large offline batches.**
7. **Use Qwen3.8 Max to analyze aggregated logs and propose patches.**
8. **Validate every patch offline before live testing.**
9. **Only add heavier methods after simple methods plateau.**

This gives the project the best chance of reaching:

    Act 1 boss 50%
    Act 2 reach 50%
    Act 2 boss 50%
    Final boss reach 10%
    Final boss beat 10%

while keeping the system maintainable across STS2 updates.

---

# 28. References / Research Basis

The following are broad references supporting the methods used in this proposal. These are not necessarily STS2-specific, but they support the chosen architecture.

1. Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction*.  
   Foundational RL concepts for sequential decision-making.

2. Coulom, R. *Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search*.  
   Early MCTS methodology.

3. Kocsis, L., & Szepesvári, C. *Bandit based Monte-Carlo Planning*.  
   UCT and search under uncertainty.

4. Browne, C. B., et al. *A Survey of Monte Carlo Tree Search*.  
   Overview of MCTS strengths and limitations.

5. Silver, D., et al. *Mastering the game of Go with deep neural networks and tree search*.  
   Combining learned evaluation with search.

6. Silver, D., et al. *Mastering the game of Go without human knowledge*.  
   Self-play and learned policy/value systems.

7. Schrittwieser, J., et al. *Mastering Atari, Go, chess and shogi by planning with a learned model*.  
   Planning with learned models.

8. Mnih, V., et al. *Human-level control through deep reinforcement learning*.  
   Deep RL in game environments.

9. Vinyals, O., et al. *Grandmaster level in StarCraft II using multi-agent reinforcement learning*.  
   Large-scale game RL with complex decision-making.

10. OpenAI. *Dota 2 with Large-Scale Deep Reinforcement Learning*.  
   Large-scale RL in complex game environments.

11. Levine, S., et al. *Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems*.  
   Learning from logged data without live interaction.

12. Fujimoto, S., et al. *Off-Policy Deep Reinforcement Learning without Exploration*.  
   Offline RL methods for logged datasets.

13. Snoek, J., Larochelle, H., & Adams, R. P. *Practical Bayesian Optimization of Machine Learning Algorithms*.  
   Bayesian optimization for noisy expensive objectives.

14. Salimans, T., et al. *Evolution Strategies as a Scalable Alternative to Reinforcement Learning*.  
   Evolutionary optimization for policy parameters.

15. Sculley, D., et al. *Hidden Technical Debt in Machine Learning Systems*.  
   Importance of maintainability, monitoring, and debt control.

16. Amershi, S., et al. *Software Engineering for Machine Learning: A Case Study*.  
   Practical ML engineering discipline.

17. Kohavi, R., Longbotham, R., Sommerfield, D., & Henne, R. *Controlled experiments on the web: survey and practical guide*.  
   A/B testing and statistical validation.

18. Millington, I., & Funge, J. *Programming Game AI by Example*.  
   Practical game AI techniques including utility-style decision-making.

19. Game AI Pro series.  
   Utility systems, behavior trees, and practical game AI architecture.