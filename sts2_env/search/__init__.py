"""Decision-time search over the simulator.

The agent's other half. Where `gym_env` learns a policy offline and reads a
flattened vector at play time, this consults the simulator itself while deciding:
clone the state, try the lines, keep the best one.

That is affordable because the pieces are cheap. A `CombatState` deep-copies in
0.4 ms, a whole combat plays out randomly in 3 ms, and every legal card-play
sequence for one turn is 49-469 of them. The live game spends 2.6-3.6 seconds per
decision on animation regardless, so the thinking is free.

It also holds no game numbers of its own. Mega Crit rebalances Bash and a trained
policy is silently wrong until retrained; search reads the new number from the
simulator on the next decision. What has to stay correct is the simulator, which
is what the parity suites are already for.
"""
