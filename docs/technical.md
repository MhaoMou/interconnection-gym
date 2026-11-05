# Technical details

The other pages name methods in passing: support-function projection, lazy
constraints, water-filling by LPs, exploitability, and so on. This page
explains each of them. Every entry starts with a short plain-language summary,
then gives the method, and ends with where it lives in the code.

Notation: `n` applicants, served loads `s` (MW, one entry per applicant),
requests `r`, value per MW `V`, commitment per MW `kappa`, minimum allocation
`beta`. A region is the set of served loads the network can host.

## Batch regions from a DC network

### Downward-closed regions and hosting capacity

**In short.** A region is downward closed if lowering any applicant's load
never makes a feasible point infeasible. The theory assumes this. An explicit
region is the polytope `{s >= 0 : A s <= b}`; it is downward closed exactly
when every entry of `A` is nonnegative. Each row is a shared limit (a branch,
spare generation), and `A[c, i]` is how much one MW of applicant `i` loads
limit `c`.

**Method.** `Region` rejects negative entries of `A` and negative `b`, and
rejects any applicant that loads no limit (its load would be unbounded). The
single-bus hosting capacity is the most applicant `i` can be served with
everyone else at zero: `h_i = min over c with A[c, i] > 0 of b_c / A[c, i]`.
The planner's optimum is one LP, `max sum(s)` over the region.

**Code.** `icgym.region.Region`: `hosting_capacities`, `max_total`,
`contains`, `restrict`, and `Region.downward_closed`.

### The period problem in PTDF form

**In short.** One period (one hour of a representative day) can host new
loads `x` at the batch's buses if the existing generators can be redispatched
to serve the base load plus `x`, with every branch flow within its limit. This
is a lossless DC optimal power flow (DCOPF) written with power transfer
distribution factors (PTDFs).

**Method.** `DCNetwork.from_csv_dir` builds the PTDF matrix from the in-service
branches: with incidence matrix `C` and branch susceptances `1/x_k`, the bus
susceptance matrix is `B = C' diag(1/x) C`. Removing the reference bus (type 3)
gives `PTDF = diag(1/x) C_r inv(B_rr)`, with a zero column for the reference
bus. `PTDF[k, j]` is the flow on branch `k` per MW injected at bus `j` and
withdrawn at the reference bus. In period `t`, with generator outputs `p` and
new loads `x >= 0`:

- power balance: `sum(p) - sum(x) = total demand in t`;
- branch limits: `|PTDF_gen p - PTDF_site x - PTDF d_t| <= limit`, where `d_t`
  is the base demand at every bus;
- generator bounds: `p_min <= p <= p_max`.

Losses, reactive power, and voltages are ignored (see
[AC power flow](#ac-power-flow) for the full model). Active DC lines are not
supported and raise an error.

**Code.** `icgym.network.DCNetwork` (`from_csv_dir`, `period_problem`) and
`icgym.network.PeriodProblem`.

### Support LPs

**In short.** The support function of a region in direction `d` is the
largest value of `d'x` over the region. It is one LP, and its solution is a
point on the region's boundary.

**Method.** `PeriodProblem.support(d)` solves `max d'x` over `x >= 0`, the
redispatch `p`, and the period's constraints, and returns the value and the
maximizing `x`. Some coordinates can be held at zero (`zero=`). LPs are solved
by HiGHS through `scipy.optimize.linprog`. On numerical trouble the objective
is rescaled and the LP retried with the dual simplex, then the interior-point
method. An infeasible LP means the base case itself cannot be served; an
unbounded one means some bus loads no limit in this period.

**Code.** `icgym.network.PeriodProblem.support`, `icgym.network._solve_lp`.

### Support-function projection

**In short.** The exporter needs the facets (the inequalities `g'x <= h`) of
each period's region in the space of the new loads only. The region is the
projection of a higher-dimensional polytope (loads plus every generator's
output), and projecting by eliminating the generator variables is expensive.
Instead, the exporter grows an inner approximation from boundary points found
by support LPs, until every facet of that approximation is confirmed to be a
facet of the true region.

**Method.** For one period, `period_facets`:

1. Computes each bus's hosting capacity with a support LP in the coordinate
   direction. Buses with zero capacity are fixed by a row `x_i <= 0` and
   dropped. With one bus left, the region is the interval `[0, h]`.
2. Seeds a point set with feasible points only: the origin, each single-bus
   capacity point (other loads at zero), the maximizer of each coordinate
   direction, and the maximizer of the all-ones direction.
3. Takes the convex hull of the points (Qhull, through
   `scipy.spatial.ConvexHull`). Because every point is feasible and the region
   is convex, the hull lies inside the region.
4. For each hull facet with unit normal `g` and offset `h` (skipping the
   facets `x_i >= 0`), solves the support LP in direction `g`. If the support
   exceeds `h` by more than `tol` times the largest hosting capacity, the
   maximizer is a new boundary point outside the hull, and it is added. If
   not, the facet is confirmed: no feasible point lies beyond it, so it is a
   facet of the true region.
5. Repeats from step 3 until a pass adds no point (at most `max_iter = 200`
   passes).

Each pass costs one LP per unconfirmed hull facet, and confirmed facets are
not tested again. The number of facets grows fast with the number of buses,
which is why explicit regions are practical up to about three applicants on a
large network. With a relative `gap`, the loop stops once no support exceeds
a facet by more than `gap` times the largest capacity; the result is then the
hull itself, an inner approximation of the region. Periods are independent, so
`export_region(..., workers=k)` runs them in a process pool.

**Code.** `icgym.network.period_facets`, `icgym.network.export_region`.

### Intersecting periods and pruning facets

**In short.** A batch must be hostable in every period, so the batch region is
the intersection of the period regions. Stacking every period's facets gives
many duplicate and redundant rows, which are removed.

**Method.** `assemble_region` stacks the facets of all periods. Rows are
normalized by their right-hand side and rounded to 9 digits to drop
duplicates. Then each row is tested for redundancy with one LP: maximize the
row over `x >= 0` subject to all other kept rows. If the maximum does not
exceed the row's bound (within a small tolerance), the row is implied by the
others and is dropped. This costs one LP per row.

**Code.** `icgym.network.assemble_region`, `_prune_signed`, `_prune`.

### Largest downward-closed subset and counterflow

**In short.** On a real network, one new load can relieve a branch that
another loads, because their flows on that branch go in opposite directions.
This relief is counterflow. It appears as a negative coefficient in a facet,
and it makes the region not downward closed: reducing one load can remove
relief the other relied on. The exporter drops the relief by keeping only the
positive part of each facet. That treats counterflow as non-firm: no applicant
is promised capacity that exists only while another applicant is connected.

**Method.** With the exact signed facets `G x <= h`, the returned region is
`{x >= 0 : max(G, 0) x <= h}`, pruned again. Clipping can only tighten a
row (each negative term was lowering the left side), so the result is a
subset of the exact region, and it is downward closed. A row involving one bus
alone is unchanged, so every single-bus hosting capacity is kept. The paper
shows this is the largest downward-closed subset (Lemma 4.1).
`ExportedRegion` keeps the exact signed facets (`signed_A`, `signed_b`) so
counterflow can be inspected; `has_counterflow` reports whether any exact
coefficient is negative.

**Code.** `icgym.network.ExportedRegion`, `icgym.region.Region.downward_closed`.

## Implicit regions and lazy constraints

### Lazy branch limits

**In short.** A 2,000-bus network has thousands of branches, but only a few
limit any given load pattern. Lazy constraint generation solves the LP with
the branches known to matter, checks all the others at the solution, and adds
only those that are violated.

**Method.** Each `PeriodProblem` keeps an active set of branches (empty at
first). Each solve uses only the active rows; afterwards every branch flow is
computed from the solution, and any branch over its limit by more than
`1e-7 * max(1, largest limit)` MW joins the active set. The LP is re-solved
until no branch is violated. The result is the exact optimum: the reduced LP
is a relaxation of the full one, and its optimum satisfies every constraint of
the full one. The active set persists on the period, so later queries start
with the branches found so far.

**Code.** `PeriodProblem.support` (`icgym.network`), and the same loop in
`DCOPFRegion._optimize`, `DCOPFRegion._feasible_lazy`, and `icgym.prices.lmp`.

### Lazy periods

**In short.** A region over 120 periods is the intersection of 120 period
regions, but only a few periods (usually the peak hours) bind. `DCOPFRegion`
solves each query with the periods found binding so far and adds a period
only when the answer violates it.

**Method.** `DCOPFRegion._optimize` solves one joint LP over `y = (x, t)` (the
new loads and one auxiliary scalar used by the queries below) and a separate
generator redispatch for each included period. It starts with the three
periods of highest total demand (`initial_periods=3`). After each solve:

1. In the included periods, violated branches are added (lazy branch
   limits), and the LP is re-solved.
2. Every period left out is checked with a feasibility LP at the solution `x`
   (again with lazy branch rows). Periods that cannot host `x` are added, and
   the LP is re-solved.
3. When no branch and no period was added, the solution is optimal for the
   full multi-period problem.

The binding periods and each period's active branches persist across queries,
so later queries are cheaper. Each LP is retried with the dual simplex and
interior point if HiGHS reports numerical trouble. `lp_count` counts the LPs
solved.

**Code.** `icgym.dcopf_region.DCOPFRegion` (`_optimize`, `_joint_lp`,
`_feasible_lazy`).

### Queries on an implicit region

**In short.** `DCOPFRegion` never lists facets. Each question a rule asks is
an LP over the served loads and every period's redispatch.

**Method.** All queries call `_optimize` with a different objective and extra
constraints on `y = (x, t)`:

| Query | LP |
|---|---|
| `hosting_capacities()` | `max x_i` with the other loads fixed at zero; one LP per applicant, cached |
| `max_total()` | `max sum(x)` (the planner's optimum) |
| `max_common_fraction(r)` | `max t <= 1` with `x = t r` (the proportional rule) |
| `utilitarian(r)` | two stages (see [the utilitarian rule](#utilitarian-rule-and-its-tie-break)) |
| `water_fill(r)` | a level LP and blocking tests per stage (see [water-filling by LPs](#water-filling-by-lps)) |
| `contains(s)` | a feasibility LP per period at `s * (1 - 1e-6)` |

A uniform allocation of a few applicants on ACTIVSg2000 takes seconds, against
microseconds on an exported region. The rule `cut_order='largest_loading'`
needs explicit limit loadings and is not available on an implicit region.

**Code.** `icgym.dcopf_region.DCOPFRegion`.

## Allocation rules as optimization

### Water-filling and the max-min fair point

**In short.** The uniform rule raises every applicant's served load at the
same rate, like water rising in connected tanks. An applicant stops when it
reaches its request or when a limit it loads becomes full. With unbounded
requests the result is the max-min fair point `w`: the allocation whose
smallest share is as large as possible, then the second smallest, and so on.

**Method.** A common level `t` rises, and each rising applicant is served
`lambda_i t` (weights `lambda`, all one for the uniform rule). On an explicit
region each stage is computed in closed form:

1. For each limit, the level at which it fills is `(b_c - load of stopped
   applicants) / (sum of A[c, i] lambda_i over rising i)`.
2. The stage level is the smallest of these levels and of the request levels
   `r_i / lambda_i`.
3. Applicants that reached their request stop, and so do all rising
   applicants loading a limit that just filled.

At least one applicant stops per stage, so there are at most `n` stages. Each
stage is recorded as `Stage(level, applicants, limits)`; these are the levels
`tau_k` and blocked sets `B_k` of the theory, and their number `K` sets the
`2K` bound on best-response rounds. Comparisons use a relative `1e-12` plus
an absolute `1e-9` (scaled by the largest `b` for limits).

**Code.** `icgym.waterfill.water_fill`, `max_min_fair_point`;
`icgym.oracles.stages`.

### Water-filling by LPs

**In short.** On an implicit region there are no rows to compute fill levels
from. Each stage instead finds the highest common level by an LP, then asks,
applicant by applicant, whether it could still grow alone.

**Method.** At each stage of `DCOPFRegion.water_fill`:

1. **Level LP.** Maximize `t` subject to `x_i = lambda_i t` for the rising
   applicants, stopped applicants fixed at their served loads, and
   `t <= min r_i / lambda_i`. The rising applicants are set to
   `lambda_i t`.
2. **Request test.** An applicant within `1e-7 * scale` of its request stops
   (`scale` is the largest hosting capacity, at least 1).
3. **Blocking test.** For every other rising applicant `i`, maximize `x_i`
   with every other load fixed. If the maximum exceeds the current `s_i` by at
   most `1e-6 * scale` MW, `i` is blocked and stops.

On a convex region at least one applicant is blocked at the level LP's
optimum, so each stage stops someone. A stage costs one level LP plus one LP
per rising applicant, and there are at most `n` stages, so a full allocation
costs `O(n^2)` LPs, each with lazy periods and branches.

**Code.** `icgym.dcopf_region.DCOPFRegion.water_fill`.

### Proportional rule: the common fraction

**In short.** The proportional rule serves every request scaled by one common
fraction `alpha <= 1`, the largest that fits.

**Method.** On an explicit region,
`alpha = min(1, min over c of b_c / (A r)_c)`. On an implicit region it is one
LP: maximize `t <= 1` with `x = t r` feasible.

**Code.** `icgym.rules.ProportionalRule`, `DCOPFRegion.max_common_fraction`.

### Utilitarian rule and its tie-break

**In short.** The utilitarian rule serves the largest total load with no one
served above its request. The maximizer is usually not unique, so a second LP
picks, among the maximizers, the one whose smallest service fraction `s_i/r_i`
is largest.

**Method.** Stage 1: `max sum(s)` subject to the region and `0 <= s <= r`,
giving the total `T`. Stage 2: maximize `z <= 1` subject to the region,
`0 <= s <= r`, `s_i >= z r_i` for every positive request, and
`sum(s) >= T` minus a small relative slack (`1e-9` on explicit regions, `1e-7`
on implicit ones). If stage 2 fails, the stage-1 solution is returned.

**Code.** `icgym.rules.UtilitarianRule`, `DCOPFRegion.utilitarian`.

### Minimum allocation and cut order

**In short.** With a minimum allocation `beta`, an applicant offered less
than `beta` of its request is dropped from the batch, and the rule is run
again for the rest. Only one applicant is dropped at a time; the cut order
says which.

**Method.** `AllocationRule.allocate` loops:

1. Allocate among the active applicants.
2. Mark as short every active applicant with `s_i < beta r_i` (with a relative
   slack of `1e-9`). If none is short, stop.
3. Drop one short applicant and go back to step 1.

The cut orders:

- `largest_request` (default for the uniform and utilitarian rules): drop the
  short applicant with the largest request.
- `largest_loading` (default for the proportional rule): find the most loaded
  limit, measured by `(A r)_c / b_c` over the short and served applicants, and
  drop the short applicant with the largest `A[c, i] r_i` on it. Dropping by
  MW instead can break the proportional rule's equilibrium.

Ties go to the lowest index. There are at most `n` cuts, each a full
re-allocation.

**Code.** `icgym.rules.AllocationRule` (`allocate`, `_next_cut`).

### Batched allocation

**In short.** Learning needs many allocations. `allocate_batch` runs the
uniform and proportional rules on `E` request profiles at once with numpy.

**Method.** The same stages as above, computed on `(E, n)` arrays, with the
minimum-allocation loop applied row by row in vector form. Other rules and
implicit regions fall back to a Python loop.

**Code.** `icgym.vector.allocate_batch`, `water_fill_batch`.

## Equilibria and how they are found

### Exploitability

**In short.** Exploitability measures how far a profile is from an
equilibrium: each applicant's largest payoff gain from changing its own
strategy while the others keep theirs. It is zero at a pure equilibrium.

**Method.** For each applicant, `max(0, best-response payoff - current
payoff)`, with the best response from the search below. The benchmarks divide
it by `V_i h_i`, the value of the applicant's hosting capacity. It is only as
good as the search: a positive number is a real deviation found, while a zero
means the search found none.

**Code.** `BatchGame.exploitability`, `TwoPartGame.exploitability`,
`LocationGame.exploitability`.

### Best-response search

**In short.** A best response is the request that maximizes one applicant's
payoff with the others fixed. Under the uniform rule without a minimum
allocation it has a closed form. Elsewhere the payoff can jump (an applicant
is cut, or overtakes a rival in the cut order), so it is found by a grid
search with local refinement.

**Method.** `BatchGame.best_response`:

- **Closed form** (uniform rule, `beta = 0`, `0 < kappa < V_i`, no forfeit):
  water-fill with applicant `i`'s request set to infinity, and request exactly
  what that serves it.
- **Search** otherwise: evaluate the payoff on `grid` points (default 400) of
  `[0, upper]`, plus the current request, the hosting capacity, and the
  "kinks": the requests where the requested load on a limit reaches
  `b_c / beta`, where
  applicant `i` alone fills a limit, and where its request or loading ties a
  rival's, each also tried `1e-9` either side. The four best candidates are
  refined by bounded scalar minimization within one grid step. The upper end
  is the largest of `h_i`, `h_i / beta`, the request that overtakes every
  rival in the cut order, and the largest current request, capped at
  `20 h_i`, plus 5%.

Under a minimum allocation the supremum may not be attained (a best response
can be a vanishing undercut of a rival); the search returns the best point
found. The two-part games (`GenerationGame`, `NonFirmGame` and their network
versions) search a two-dimensional grid of request and second part, then
refine the two coordinates in turn.

**Code.** `BatchGame.best_response`, `_exact_best_response`, `_kinks`,
`_search_upper`; `TwoPartGame.best_response` (`icgym.devices`).

### Best-response dynamics

**In short.** Applicants take turns best-responding. If the profile stops
changing, it is a pure equilibrium.

**Method.** In each round every applicant best-responds once, in index order
(`order='cyclic'`) or a random permutation. A round settles only if no request
moved by more than `tol` and no applicant could gain more than `gain_tol`
(default `1e-9 * max |V_i h_i|`). The second test matters under a minimum
allocation, where a profile can stand still while a large gain remains.
Returns the profile, the history, and the number of rounds with a change, or
`None` if it did not settle within `max_rounds`.

**Code.** `BatchGame.best_response_dynamics`,
`TwoPartGame.best_response_dynamics`, `LocationGame.best_response_dynamics`.

### Equilibria on strategy grids

**In short.** When there is no closed form and the search is expensive (as on
a `DeviceRegion`), `grid_equilibria` restricts every applicant to a finite
list of strategies and finds every pure equilibrium of that finite game
exactly.

**Method.** It enumerates every profile in the product of the strategy sets.
A profile is an equilibrium if no applicant gains more than `tol` by switching
to another strategy in its own set. Payoffs are cached by index profile, so
the cost is at most `prod |S_i|` payoff evaluations, each one allocation. The
result is exact for the restricted game only: the grid must contain the
levels where an equilibrium can sit, or it is missed, and a grid profile can
be an equilibrium only because better deviations lie off the grid.

**Code.** `icgym.game.grid_equilibria`.

### Equilibrium oracles

**In short.** The oracles return what the paper proves for a game, so learned
or searched behavior can be scored against it. They check the game's rule,
fee, and region and either return the answer or `{"known": False}`.

**Method.** `predicted_equilibrium(game)` for `BatchGame`:

- uniform rule (optionally weighted): compute `w`, the (weighted) max-min fair
  point;
    - with a `FlatFee`: the band from `w` to `w / beta` (`kind='band'`);
    - with `0 < kappa < V` and no forfeit: `w` (`kind='unique'`), or
      `kind='epsilon'` when the fee includes an energy charge;
- proportional rule on a single limit, equal values, `kappa > 0`:
  `r_i = rho b / (n a_i)` with
  `rho = min(max((1 - 1/n) V / kappa, 1), 1/beta)`;
- otherwise unknown.

The extension games have their own `predicted_equilibrium` methods
(`GenerationGame`, `NonFirmGame`, `NetworkNonFirmGame` with the two-tier
rule). `NetworkGenerationGame` returns unknown, because on a real network the
capacity a generator frees varies with load; use grids there.

**Code.** `icgym.oracles.predicted_equilibrium`, `proportional_inflation`.

### Price of anarchy

**In short.** The price of anarchy compares the planner's best total with the
total served at equilibrium.

**Method.** `price_of_anarchy(region, s) = max_total / sum(s)`. The theory's
bound for the uniform rule is `(1 + nR)^2 / (4nR)`, with `R` the ratio of the
largest to the smallest single-bus hosting capacity (`poa_bound`).

**Code.** `icgym.oracles.price_of_anarchy`, `poa_bound`.

### Improving moves in the location game

**In short.** In the location game each applicant picks a placement, and its
payoff is its share of the max-min fair point of the resulting region. The
theory says every improving move raises the sorted vector of shares, so
best-response dynamics cannot cycle. `improving_moves` checks this on every
profile.

**Method.** Shares are computed once per multiset of placements (applicants
at the same placement are interchangeable) and cached. A move counts as
improving only if it raises the mover's share by more than `tol` (default
`1e-4`, in MW for network regions), which keeps solver noise out of the
counts. For each improving move, the sorted shares before and after are
compared at the first entry that differs by more than `tol`; a decrease
there, or no such entry, is a violation. `pure_equilibria` enumerates every
profile. On a network, `LocationGame.from_network` builds one `DCOPFRegion`
per new multiset.

**Code.** `icgym.location.LocationGame` (`shares`, `potential`,
`improving_moves`, `pure_equilibria`).

### Learning baselines

**In short.** Three reference learners for the benchmark tasks.

**Method.**

- `best_response`: best-response dynamics from a random start.
- `hedge`: multiplicative weights on a grid of actions (41 request levels; 17
  times 5 for two-part actions; the menu for placements). Each round every applicant's weights move by
  `eta` times the average payoff of each grid action against sampled rival
  actions (full information); the most likely action is returned.
- `independent_pg`: each applicant has a Gaussian policy in logit space
  (actions `sigmoid(z)`), updated from its own rewards only, with normalized
  advantages and a linearly decaying step; a softmax over the menu for
  placements.

**Code.** `icgym.benchmarks` (`best_response`, `hedge`, `independent_pg`).

## Prices and devices

### LMPs from LP duals

**In short.** The locational marginal price (LMP) at a bus is the cost of
serving one more MW there. It equals the energy price plus the cost of the
congestion that MW adds, and both come from the dual variables of the
least-cost dispatch LP.

**Method.** `lmp` solves the least-cost redispatch of one period with the new
loads fixed, minimizing `gen_cost' p` subject to power balance, generator
bounds, and branch limits added lazily. The price at bus `k` is the dual of
the balance constraint plus, for each active branch, the difference of the
duals of its upper and lower flow limits times `PTDF[branch, k]`. Generator
costs come from `marginal_cost_c1_per_mwh` in `generator_limits.csv`.

**Code.** `icgym.prices.lmp`.

### Energy charge and the epsilon-equilibrium

**In short.** `EnergyCharge` bills each applicant its served energy at its
own bus's LMP. Its own load moves that price, so the uniform rule's `w` is no
longer an exact equilibrium. It is an epsilon-equilibrium: no applicant gains
more than a small epsilon by deviating, with epsilon set by how far one
applicant can move its own price.

**Method.** `cost_i = share * sum over t of w_t LMP_t(s) s_i`, with period
weights `w_t` (hours times represented days, and any annuity). Each new
served profile costs one LP per period; prices are cached by the rounded
profile, and each period's active branches are reused. `FeeSum` adds it to a
commitment. `predicted_equilibrium` then returns `kind='epsilon'`. Searching
best responses is expensive here, which is why `best_response` accepts an
`interval`.

**Code.** `icgym.prices.EnergyCharge`, `FeeSum`.

### Devices on the network

**In short.** `DeviceRegion` is the exact region when applicants also run
on-site generators, batteries, or curtailable load, dispatched hour by hour
over the network's real days. A device frees capacity only in the hours it
runs, so its value follows the daily load shape.

**Method.** For each period `t` and applicant `i` the LP has the
generator output `g`, battery charge `c` and discharge `d`, and state of
charge. The net load `m[t, i] x_i - g + c - d` must be nonnegative (no export)
and must fit the period's DCOPF after redispatch. The constraints:

- **load scale** `m[t, i]`: `1 - phi_i` in the peak periods for a curtailable
  fraction `phi_i`, or `0` in the peak periods for non-firm load;
- **run-hour budget**: `sum over the day of g[t, i] <= run_hours * G_i`, with
  `0 <= g <= G_i`;
- **proportional split** (`split='proportional'`, the gross rule): in every
  period all generators run at the same fraction of their capacities, so the
  room they free is shared; `split='free'` drops this;
- **storage cycles**: `0 <= c, d <= P_i`, state of charge between 0 and
  `storage_hours * P_i`, and `soc[t+1] = soc[t] + c[t] - d[t]` around each day,
  with the last hour wrapping to the first, so every day ends where it began
  (no round-trip losses are modeled).

Devices couple the hours of a day, so whole days enter lazily (starting with
the peak day), and branches enter lazily within each period, as in
`DCOPFRegion`. `dispatch(s)` finds, day by day, the dispatch that serves `s`
with the least weighted generator output, which gives each owner's fuel
energy. `peak_periods` and `curtailment_scale` build the load scale.

**Code.** `icgym.network_devices.DeviceRegion`, `peak_periods`,
`curtailment_scale`.

### Gross rule and crediting

**In short.** Who keeps the capacity a device frees is the operator's choice.
Under the gross rule the freed room joins the common pool and is shared.
Under crediting it goes to the device's owner.

**Method.** In `NetworkGenerationGame` (strategies `(r_i, G_i)` with
`G_i <= gamma r_i`):

- `'gross'`: the uniform rule on the `DeviceRegion` with these generators and
  the proportional split;
- `'credit'`: the extra total load all generators unlock together,
  `max_total(G) - max_total(0)`, is split among the owners in proportion to
  `G_i`; the uniform rule then serves the requests net of these credits on the
  region without generators, and `s_i = min(r_i, credit_i + share_i)`.

`unlocking_ratio` measures the MW of hosting capacity one MW of generator
frees for one applicant alone (`theta`). On the explicit peak-period model
(`GenerationGame`), `theta = min(1, run_hours / (nu dt))` for `nu` peak periods
of `dt` hours, and the gross rule is water-filling on
`{s >= 0 : A (s - theta G)^+ <= b}`, solved exactly by a breakpoint search in
`water_fill_offset`.

**Code.** `icgym.network_devices.NetworkGenerationGame`;
`icgym.devices.GenerationGame`, `Device`, `peak_theta`, `water_fill_offset`.

### Non-firm service on the network

**In short.** Firm load is served in every hour; non-firm load only off peak.
The two-tier rule allocates the two tiers one after the other.

**Method.** In `NetworkNonFirmGame` with `rule='two_tier'`, firm requests are
water-filled on the region over all periods. The firm loads are then added to
the base demand, backed off by a relative `1e-9` so the binding period stays
feasible, and non-firm requests are water-filled on the off-peak region that
remains (load scale zero in the peak periods). With `rule='gross'`, each
applicant picks a request and a curtailable fraction `phi_i <= gamma`, and one
common level is water-filled on the region with net loads `(1 - phi_i) s_i` in
the peak periods.

**Code.** `icgym.network_devices.NetworkNonFirmGame`, `with_added_load`;
`icgym.devices.NonFirmGame` for explicit regions.

## Sequential batches and the operator

### Residual region

**In short.** After a batch, served MW connect for good and use up capacity.
The next batch sees only what is left.

**Method.** `SequentialBatchGame` works on a region over buses (column `j` is
one MW at bus `j`). The residual capacity of limit `c` is
`b_c - (A connected)_c - growth_c * k` after `k` batches, floored at zero. The
batch region takes the bus column of each queued applicant and drops limits
none of them loads. Served loads are also capped at each applicant's remaining
need.

**Code.** `icgym.sequential.SequentialBatchGame` (`residual_b`,
`batch_region`).

### Myopic batch equilibrium

**In short.** The ground truth for each batch is the equilibrium of that
batch alone: water-filling the residual region, with each applicant's request
capped at its remaining need.

**Method.** `myopic_equilibrium` returns `water_fill(batch_region, need)`. It
ignores the effect of today's request on later batches, which is why it is
called myopic.

**Code.** `SequentialBatchGame.myopic_equilibrium`.

### Equilibrium response

**In short.** When the operator searches over designs, the applicants must
answer its choice of rule, `beta`, and `kappa` strategically.
`equilibrium_response` computes their answer.

**Method.**

1. Applicants whose value is at most `kappa` request nothing.
2. Each remaining applicant's need is added as an extra limit `s_i <= need_i`.
3. `w` is water-filling on this capped region.
4. Unweighted uniform rule: the answer is `w`; with a flat fee (or
   `kappa = 0`) and `flat_fee_end='high'`, the band's high end `w / beta`.
5. Any other rule: best-response dynamics from `w`, with the search
   (`grid=80`, at most 15 rounds). The last profile is returned even if the
   dynamics did not settle.

The operator's reward (`Objective`) is efficiency minus a weight times the
share of requested MW not served, minus a weight times one minus Jain's
fairness index of the served shares of hosting capacity.

**Code.** `icgym.rule_design.equilibrium_response`, `Objective`,
`batch_metrics`, `OperatorEnv`, `SequentialOperatorEnv`.

## AC power flow

### The AC period problem

**In short.** `ACOPFRegion` replaces the DC model with the full AC equations,
which include losses, reactive power, and voltage limits. The region is then
not a polytope and may not be downward closed.

**Method.** In each period the variables are bus voltage angles and
magnitudes and generator active and reactive outputs, in per unit. The
constraints are active and reactive power balance at every bus (with branch
two-port admittances, taps, phase shifts, and shunts), apparent-power limits
`P^2 + Q^2 <= rate^2` at both ends of every rated branch, voltage limits,
generator limits, and a zero reference angle. New loads draw reactive power
`tan(phi) x_i` with `tan(phi) = sqrt(1 - pf^2) / pf` for a lagging power
factor `pf`. The programs are built symbolically in CasADi and solved by Ipopt
(tolerance `1e-9`, at most 3000 iterations); one solver is compiled per
problem shape and reused.

**Code.** `icgym.ac_region.ACNetwork`, `_Period`, `ACOPFRegion._joint_solver`.

### Local optima and restarts

**In short.** Ipopt finds local optima of nonconvex programs, so a maximum it
returns can fall short of the true one. Starting points matter, and the region
can try several.

**Method.** Each solve tries, in order, the previous solution of each period
(a warm start) and a flat start (angles zero, voltages one, outputs at the
middle of their ranges). With `multistart=k`, `k` random starts are added
(small angles, voltages near one, outputs within limits) and the best result
is kept; otherwise the first successful solve is used. Every query has a
feasible point (no new load), so if all of these fail the failure is
numerical, and up to eight more random starts are tried. Periods enter lazily
as in the DC region: up to three unhosted periods are added at a time,
highest total demand first.

**Code.** `ACOPFRegion._solve`, `_random_start`, `_optimize`.

### Feasibility checks

**In short.** Saying a point is hosted is easy to certify: Ipopt returns an
operating point that meets every limit. Saying it is not hosted is not, since
Ipopt may just have failed to find such a point.

**Method.** To test whether period `t` hosts `x`, the region maximizes
`a <= 1` with loads `a x` hosted. A solve that reaches `a >= 1 - 1e-7` proves
the point is hosted. "Not hosted" is returned only if both the warm and the
flat start were tried, at least one solved, and neither reached 1. Random
restarts are tried only if neither of those two solves succeeded.
`contains(s)` runs this test in every period at `s * (1 - 1e-6)`, starting
with the binding periods and then in order of peak load, so that a point that
is not hosted usually fails within a few solves. `operating_point(s, t)` returns the voltages, angles,
outputs, and branch MVA flows that host `s`.

**Code.** `ACOPFRegion._feasible_lazy`, `contains`, `operating_point`.

### Certified water-filling

**In short.** Water-filling on the AC region uses the same stages as on the DC
region, but each step is checked with feasibility tests, which are more
reliable than the maxima.

**Method.** At each stage:

1. The level comes from the level program, as in the DC case. If Ipopt fails
   on it even after random restarts, the level is found instead by bisection
   on hosted points along the ray (to a relative `1e-4`), and the stage is
   counted in `bisected_stages`.
2. The level is pushed: while the point a relative step `certify` (default
   `1e-3`) further along the ray is hosted, the level moves there and the
   level program is re-solved from it (at most 50 pushes).
3. A rising applicant counts as blocked only if growing it alone by the
   relative step `certify` is not hosted.

If no applicant is blocked although the common level cannot rise (possible
when one load relieves another), the lowest-indexed applicant with room is
frozen, and the stage is recorded in `unblocked_stages`.

**Code.** `ACOPFRegion.water_fill`, `_bisect_level`, `_unblocked_stage`.

### Downward-closure check

**In short.** The theory's guarantees need a downward-closed region. On the
AC region that can fail: removing load can raise voltages above their limits.
`downward_closure_check` tests the assumption on sample points of the
boundary.

**Method.** Each sample is a boundary point: water-filling with random weights
(uniform on `[0.2, 1]`) and requests of twice the hosting capacities. For each
applicant and each factor in `cuts` (default 0 and 0.5), its load is multiplied
by that factor and the point is tested with `contains`. The result lists the
number of tests and the violations. Passing is evidence, not a proof; a
violation is a real counterexample, up to the reliability of the
feasibility test.

**Code.** `icgym.ac_region.downward_closure_check`.
