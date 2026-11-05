# Regions from a network

## Exporting a batch region

`icgym.network` builds the batch region of a set of buses from a DC
power-flow scenario. The scenario is a folder of CSV files: buses, branches,
generators, and hourly demand. A period can host new loads `x` if some
redispatch of the existing generators serves them within branch and
generator limits.

The exporter works in three steps:
1. It recovers each period's region exactly, by [support-function projection](technical.md#support-function-projection).
   Branch limits are [added lazily](technical.md#lazy-branch-limits), so 2,000-bus networks stay fast.
2. It intersects the periods.
3. It keeps the positive parts of the facets. That is the [largest
   downward-closed subset](technical.md#largest-downward-closed-subset-and-counterflow), which treats counterflow between new loads as
   non-firm.

The exact signed facets are stored alongside, so counterflow can be
inspected.

```bash
icgym-export --data path/to/scenario --buses 9,13,14 --workers 8 --out region.json
```

```python
from icgym import load_region_record
record = load_region_record("region.json")   # region, signed facets, metadata (input hashes, periods)
```

Scenario files:

| File | Columns used |
|---|---|
| `bus.csv` | `bus_id`, `type` (3 = reference bus) |
| `branch_limits.csv` | `branch_id`, `from_bus_id`, `to_bus_id`, `reactance_pu`, `thermal_limit_mva`, `status` |
| `generator_limits.csv` | `generator_id`, `bus_id`, `p_min_mw`, `p_max_mw`, `status` |
| `demand_bus.csv` | `period` and one column per bus (MW) |
| `dcline.csv` | `status`, `from_bus_id`, `to_bus_id`, `Pt` (active DC lines are not supported yet) |

## Large batches: `DCOPFRegion`

Explicit regions are practical up to about three applicants on a large
network. Beyond that, the per-period regions have thousands of facets.

For larger batches, `DCOPFRegion` keeps the region implicit:
- It answers every query with LPs over the served loads and each period's
  redispatch: [water-filling](technical.md#water-filling-by-lps), hosting capacities, the planner's optimum, and
  the proportional and utilitarian rules.
- It adds [binding periods](technical.md#lazy-periods) and branch limits lazily.
- It plugs into the same rules, games, and environments, at seconds per
  allocation.

```python
from icgym import BatchGame, DCOPFRegion, UniformRule
from icgym.network import DCNetwork

net = DCNetwork.from_csv_dir("path/to/activsg2000_scenario")
region = DCOPFRegion(net, ["7106", "5430", "5276", "7259", "5250"])
game = BatchGame(region, UniformRule(beta=0.5))
```

`DCOPFRegion` is the exact network region, so it counts the relief that one
new load gives another ([counterflow](technical.md#largest-downward-closed-subset-and-counterflow)). The exported `Region` is its largest
downward-closed subset, which the theory assumes. The two agree wherever
counterflow does not bind.

`LocationGame.from_network(net, menu, n)` builds a location game on any DC
network, using exact regions.

## AC power flow: `ACOPFRegion`

The DC model ignores losses, reactive power, and voltage limits. `ACOPFRegion`
answers the same queries under the [full AC power-flow equations](technical.md#the-ac-period-problem):
- A period hosts the new loads only if some redispatch of the existing
  generators, in active and reactive power, keeps every branch within its
  MVA rating, every bus voltage within its limits, and every generator
  within its active and reactive limits.
- New loads draw reactive power at a fixed lagging `power_factor`.
- Periods enter lazily, as in `DCOPFRegion`.

It needs the `ac` extra, which installs CasADi with the Ipopt solver:

```bash
pip install "interconnection-gym[ac] @ git+https://github.com/MhaoMou/interconnection-gym"
```

```python
from icgym import ACOPFRegion, BatchGame, PerMWCommitment, UniformRule, downward_closure_check
from icgym.datasets import ieee14_ac_network

net = ieee14_ac_network()                       # the IEEE 14-bus case with the package's limits and days
region = ACOPFRegion(net, ["9", "13"], power_factor=0.95)
region.hosting_capacities()                     # MW of new active load at each bus alone
game = BatchGame(region, UniformRule(), PerMWCommitment(0.4), value=1.0)
downward_closure_check(region, samples=10)      # test the theory's assumption on boundary samples
```

`ACNetwork.from_matpower` reads any MATPOWER case, with optional MVA limits
and per-period load multipliers.

!!! note "What carries over from the theory"
    The theory's proofs assume a downward-closed polytope. The AC region is
    not a polytope, and it need not be downward closed (removing load can
    raise voltages above their limits). Water-filling, the rules, and the
    games still run on it, but the guarantees become checks:
    [`downward_closure_check`](technical.md#downward-closure-check) tests the assumption, and `BatchGame.exploitability`
    tests whether the max-min fair point is an equilibrium.

!!! warning "Local optima"
    Ipopt finds [local optima](technical.md#local-optima-and-restarts) of nonconvex programs. Every query starts from
    a feasible point (no new load) and maximizes a linear function of the
    loads, and the region is usually star-shaped along the directions the
    rules use, but a level can in principle fall short of the global one.
    Pass `multistart=k` to re-solve from `k` random starting points.
