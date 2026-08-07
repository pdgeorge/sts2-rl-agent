"""Every reward and penalty, in one place, so tuning is one file rather than three.

They used to live in two modules with different conventions, which made "what is
this agent actually being paid for" a question you had to answer by reading code.

HOW TO TUNE THIS

Numbers here are in units of "a won combat". COMBAT_WON is 1.0 by definition and
everything else is priced against it: an elite is worth three ordinary fights, a
boss ten, reaching a new floor about a third. Keeping one anchor at 1.0 means you
can reason about a change ("is clearing an elite really worth three fights?")
instead of juggling absolute magnitudes.

Rewards near unit scale also matter mechanically: PPO normalises advantages and
initialises its value head assuming roughly that range, so rewards in the
hundreds make the value function harder to fit for no benefit.

EVENT REWARDS, NOT POTENTIAL SHAPING (for the full run)

The run env used potential-based shaping and it went wrong in a way worth
recording. Shaping pays `gamma*phi(s') - phi(s)` every step, so when phi barely
moves it still pays `(gamma-1)*phi`, about -0.01*phi per step. Over a 400-step
run that is roughly -2.0 of pure drift, which made the reported episode return
*anti-correlated with surviving*: a 12-step run scored -1.03 and a 428-step run
scored -3.27. The learning was unaffected -- PPO optimises the discounted return,
where the shaping telescopes correctly -- but best-model selection was choosing
whichever policy died fastest.

Event rewards do not have that failure mode. They fire once, when something
actually happens. They are not policy-invariant the way potential shaping is, so
in principle they can distort what is optimal -- but floors, elites and bosses in
a roguelike are monotone, one-time events. There is no way to farm reaching
floor 7 twice.

Combat keeps its potential shaping: combats are short enough that the per-step
drift is small, and mid-fight resolution is exactly what the gap gate needs.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------

RUN_WIN = 20.0
"""Completing a run. Large, because it is the actual objective and rare."""

RUN_DEATH = -5.0
"""Dying, or hitting the step cap. Truncation pays the same as death so that
running the clock out is never a way to avoid losing."""

FLOOR_REACHED = 0.35
"""Each new floor. The main progress signal: a random policy reaches floor 3.2,
a briefly trained one about 6. Deliberately smaller than a combat win so depth
alone does not outrank playing well -- but it accumulates, so a deep run is worth
far more than a shallow one."""

COMBAT_WON = 1.0
"""One ordinary fight. The unit everything else is priced in."""

ELITE_WON = 3.0
"""An elite, on top of COMBAT_WON. Elites are optional and dangerous; taking one
on and surviving should beat routing around it."""

BOSS_WON = 10.0
"""A boss, on top of COMBAT_WON. Act-defining, and the gate to further floors."""

# ---------------------------------------------------------------------------
# Combat
# ---------------------------------------------------------------------------

COMBAT_WIN_REWARD = 1.0
COMBAT_LOSS_REWARD = -1.0

COMBAT_TRUNCATION_REWARD = -1.0
"""Equal to a loss, not worse. Making truncation worse than dying would create an
incentive to die quickly, which is a different pathology rather than a fix."""

COMBAT_HP_WEIGHT = 1.0
COMBAT_ENEMY_WEIGHT = 0.5
"""phi = HP_WEIGHT*player_hp_frac - ENEMY_WEIGHT*enemy_hp_frac, as fractions so it
stays comparable between a 40 HP hallway fight and a 500 HP boss. Player HP is
weighted double because enemy HP resets every combat and yours does not."""

COMBAT_TURN_COST = 0.005
"""Per turn, not per step -- a turn with ten card plays is not ten times worse
than one with a single play.

Charged only for the first COMBAT_FREE_TURNS. Stalling to draw the card you
need, to build block before a big hit, or to stack a power is real Slay the
Spire and has to stay affordable, so an ordinary fight pays almost nothing."""

COMBAT_FREE_TURNS = 12
"""How long a fight may run at the cheap rate before the steep one applies.

Chosen above a normal fight and below a stall. Turns beyond this are not setup
-- they are a fight that is not going anywhere."""

COMBAT_STALL_TURN_COST = 0.05
"""Per turn beyond COMBAT_FREE_TURNS. Ten times the cheap rate.

THE REQUIREMENT THIS EXISTS FOR: stalling ~20 turns must score worse than taking
damage. It did not. At a flat 0.005 a 200-turn fight cost exactly 1.0 -- the
same as COMBAT_WON -- so stalling to the cap and winning netted zero and there
was no gradient preferring an 8-turn win to an 80-turn one until the extreme.

With the progressive rate:

    10 turns  0.05   trivial, a normal fight
    20 turns  0.46   worse than losing 40% of your HP (HP_WEIGHT is 1.0 on the
                     fraction), which is the bar that was asked for
    30 turns  0.96   about a whole win
    200 turns 9.5    unambiguous

This is a watchability requirement as much as a scoring one. Cyra is watched,
and nobody wants to sit through 200 turns of a Defend loop, so a policy that
stalls is a product failure even when it eventually wins."""

COMBAT_SHAPING_SCALE = 1.0
