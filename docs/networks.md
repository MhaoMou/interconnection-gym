# Regions from a network

## Exporting a batch region

`icgym.network` builds the batch region of a set of buses from a DC
power-flow scenario. The scenario is a folder of CSV files: buses, branches,
generators, and hourly demand. A period can host new loads `x` if some
redispatch of the existing generators serves them within branch and
generator limits.

The exporter works in three steps:
1. It recovers each period's region exactly, by support-function projection.
   Branch limits are added lazily, so 2,000-bus networks stay fast.
2. It intersects the periods.
3. It keeps the positive parts of the facets. That is the largest
   downward-closed subset, which treats counterflow between new loads as
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
  redispatch: water-filling, hosting capacities, the planner's optimum, and
  the proportional and utilitarian rules.
- It adds binding periods and branch limits lazily.
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
new load gives another (counterflow). The exported `Region` is its largest
downward-closed subset, which the theory assumes. The two agree wherever
counterflow does not bind.

`LocationGame.from_network(net, menu, n)` builds a location game on any DC
network, using exact regions.
