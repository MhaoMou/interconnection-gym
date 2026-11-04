# API reference

Everything listed here is the public API of 1.x and stays stable within it.
The names a script normally needs are importable from `icgym` directly.

| Page | Modules |
|---|---|
| [Core](core.md) | regions, water-filling, rules, fees, the batch game, oracles |
| [Extensions](extensions.md) | location game; generation, storage, and non-firm service (constant-theta and dispatched over the network's days); energy at the LMP |
| [Networks and data](networks.md) | DC networks, region export, `DCOPFRegion`, bundled and on-demand data |
| [Environments](environments.md) | PettingZoo, Gymnasium, and vectorized environments |
| [Tools](tools.md) | rule comparison and benchmarks |

All the games use the same names:
- `payoffs(strategies)`;
- `payoff(i, x, strategies)`;
- `best_response(i, strategies)`, which returns the strategy and its payoff;
- `exploitability(strategies)`, which returns each applicant's gain from
  deviating;
- `best_response_dynamics(start)`, which returns
  `(strategies, history, rounds)`;
- `predicted_equilibrium()`.
