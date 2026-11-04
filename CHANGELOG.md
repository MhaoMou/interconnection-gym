# Changelog

## Unreleased (toward 1.0.0)

- **API frozen:** `tests/test_api.py` pins the public names, so a rename
  fails the tests.
- **One vocabulary for every game:** `payoffs`, `payoff`, `best_response`,
  `exploitability`, `best_response_dynamics`, and `predicted_equilibrium`.
- **Renamed:**
  - `LocationGame.gains` to `exploitability` (now in payoff units) and
    `LocationGame.dynamics` to `best_response_dynamics`;
  - `LocationGame.best_response` now returns the payoff, not the share;
  - `TwoPartGame.hosting` to `hosting_capacity`;
  - the two-part games' `X` argument and prediction key to `strategies`;
  - `fairness_worst_case` to `poa_worst_case`;
  - `LocationParallelEnv(compute_gains=)` to `compute_exploitability`, and
    `info["gains"]` to `info["exploitability"]`.
- **Added:** `BatchGame.predicted_equilibrium()`, `LocationGame.payoff`, and
  `TwoPartGame.best_response_dynamics`; `AllocationRule` and `Fee` are
  exported.
- **Devices on the network:** `DeviceRegion` dispatches generators (daily
  run-hour budget, output split by capacity), batteries (daily cycle), and
  curtailable or non-firm load hour by hour, with days and branch limits added
  lazily. `NetworkGenerationGame` (gross and credit-split rules) and
  `NetworkNonFirmGame` (two-tier and gross rules) play on it.
- **Energy at the LMP:** `lmp`, `EnergyCharge`, and `FeeSum`; the uniform
  rule's prediction is marked as an epsilon-equilibrium when prices are
  charged.
- **Sequential batches:** `SequentialBatchGame` (arrivals, residual capacity,
  growth, waiting, batch-by-batch equilibrium) and the PettingZoo
  `SequentialBatchEnv`.
- **The operator as an agent:** `OperatorEnv` and `SequentialOperatorEnv`
  (Gymnasium), `equilibrium_response`, and `Objective`.
- **Grid equilibria:** `grid_equilibria` enumerates pure equilibria on
  strategy grids, for any game. `BatchGame.best_response` takes an
  `interval`, and `DCNetwork` reads generator costs.
- **Docs:** a generated API reference, published at
  <https://mhaomou.github.io/interconnection-gym/docs/>.

## 1.0.0rc1 (2026-11-03)

This is the first release candidate.

- **Core:**
  - regions, and the uniform, proportional, and utilitarian rules, with a
    minimum allocation and a cut order;
  - per-MW, flat, and deposit fees;
  - the batch request game, best responses, and exploitability.
- **Theory oracles:**
  - the max-min fair equilibrium and the flat-fee band;
  - the proportional rule's request inflation;
  - the price-of-anarchy bound.
- **Extensions:** location game; on-site generation and storage under the
  gross and crediting rules; firm and non-firm service.
- **Networks:** an exact batch-region exporter (`icgym-export`) and the
  implicit `DCOPFRegion`.
- **Data:**
  - ACTIVSg2000 catalogs, with reference results;
  - IEEE 14-bus regions built locally from the public case file
    (`icgym.datasets`), not redistributed.
- **Environments:**
  - PettingZoo environments for requests, locations, and devices;
  - a Gymnasium single-learner environment;
  - a vectorized environment with batched allocation.
- **Tools:** rule comparison (`icgym-compare`), with a text, JSON, or HTML
  report; benchmark tasks and baselines (`icgym-bench`).
