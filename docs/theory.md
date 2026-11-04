# Model and theory

## What is modeled

| Piece | Options |
|---|---|
| Locations | **Fixed** in `BatchGame` and its environments: column `i` of the region is applicant `i` at its given bus, and the strategy is the request alone. **Chosen** from a menu of buses (or splits across buses) in `LocationGame`. |
| Region (what the grid can host) | `Region`: `{s >= 0 : A s <= b}` with `A >= 0` (downward closed). `Region.downward_closed` keeps the positive part of a general polytope, which treats counterflow between new loads as non-firm. `DCOPFRegion` is the exact network region, queried by LPs. |
| Allocation rule | `UniformRule` (the default; optional fixed `weights`), `ProportionalRule`, `UtilitarianRule`. Each has a minimum allocation `beta` and a cut order (`largest_request` or `largest_loading`). |
| Fee | `PerMWCommitment`, `FlatFee`, `SecurityDeposit` (carrying cost, forfeit on unserved MW, study fee) |
| Payoff | `V_i s_i - cost_i(r, s)`: value of the served load minus what the applicant pays |
| Location choice | `LocationGame`: placements (buses, or splits across buses) from finite menus; shares come from the request-stage equilibrium |
| On-site generation and storage | `GenerationGame` with a `Device` (a generator with a run-hour budget, or a battery); `rule='gross'` shares the freed room, `'credit'` credits it to the owner |
| Non-firm service | `NonFirmGame`: `rule='two_tier'` (firm and non-firm requests) or `'gross'` (a curtailable fraction of one common level) |

## Ground truth

These are the results of the accompanying paper that the package exposes.

- **Uniform rule:** with a per-MW commitment below the value, the max-min
  fair point `w` is the unique equilibrium, served in full, for any minimum
  allocation `beta`.
- **Best-response dynamics:** without the minimum allocation, best responses
  reach `w` within `2K <= 2n` rounds from any start, where `K` is the number
  of water-filling stages.
- **Flat fee:** every profile in `[w, w/beta]` is an equilibrium, and all of
  them serve `w`.
- **Proportional rule on one limit:** requests are inflated by
  `rho = min(max((1 - 1/n) V/kappa, 1), 1/beta)`.
- **Weighted uniform rule:** with weights `1/a_i` on one limit, it gives the
  proportional rule's service without the inflation.
- **Price of anarchy:** at most `(1 + nR)^2 / (4nR)`, where `R` is the ratio
  of the largest to the smallest single-site hosting capacity.
- **Location game:** on a downward-closed network a strong pure equilibrium
  exists. Every improving move raises the sorted shares lexicographically, so
  best-response dynamics converge.
- **On-site generation:**
  - With devices fixed, `w(G)` is the unique request equilibrium.
  - On one limit, the gross rule has three regimes, set by the builder's and
    scrapper's margins: no build, everyone at the cap, or no pure equilibrium.
  - The crediting rule has a unique equilibrium on any region.
  - Storage is a device with `theta_S = min(1, eta / (nu dt))`.
- **Non-firm service:** the two-tier rule has an equilibrium on any network,
  with every request served in full. The gross rule repeats the generator
  regimes.

## Oracles

- `predicted_equilibrium(game)` and the `predicted_equilibrium()` method of
  the extension games return the theory's answer, or `{"known": False}`.
- `price_of_anarchy`, `poa_bound`, `stages`, and `proportional_inflation`
  compute the quantities above.
- `exploitability(...)` returns each applicant's gain from its best unilateral
  deviation, found by numerical search.
- `LocationGame.improving_moves()` counts improving moves and checks that each
  one raises the sorted shares.

The tests check each oracle against the engine. They also check that the
search finds a profitable deviation where the theory says no pure
equilibrium exists.

An improving move counts only if it raises the mover's share by more than
`tol` (default 1e-4, in MW for network regions), which keeps LP-solver noise
out of the counts.
