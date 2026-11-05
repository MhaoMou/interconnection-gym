# Environments and benchmarks

## Environments

In every environment except `LocationParallelEnv`, each applicant's bus is
fixed and given by the game's region; agents choose how much to request (and,
in `DeviceParallelEnv`, a device or non-firm share). In
`LocationParallelEnv` agents choose the bus, and the request stage is played
at its equilibrium.

Install with `pip install "interconnection-gym[envs]"`.

| Class | API | Actions | Game |
|---|---|---|---|
| `BatchRequestParallelEnv` | PettingZoo `ParallelEnv` | request at the applicant's fixed bus, as a fraction of `request_cap * h_i` | `BatchGame` |
| `SingleApplicantEnv` | Gymnasium `Env` | the learner's request; rivals follow fixed policies | `BatchGame` |
| `LocationParallelEnv` | PettingZoo `ParallelEnv` | `Discrete`: a placement from the menu | `LocationGame` |
| `DeviceParallelEnv` | PettingZoo `ParallelEnv` | `[0, 1]^2`: request, plus device size or non-firm request | `GenerationGame`, `NonFirmGame` |
| `icgym.vector.VectorBatchRequestEnv` | numpy arrays, `E` copies per call | `(E, n)` | `BatchGame` |

- **Rewards:** payoffs divided by the value at the hosting capacity, `V_i h_i`.
- **`info`:** carries the theory's prediction and, on request, each
  applicant's exploitability.
- **Vectorized environment:** [`allocate_batch`](technical.md#batched-allocation) vectorizes the uniform and
  proportional rules, about 50 times faster than a Python loop.

```python
from icgym import BatchGame, PerMWCommitment, UniformRule, ieee14
from icgym.envs import BatchRequestParallelEnv

game = BatchGame(ieee14((9, 13, 14)), UniformRule(beta=0.5), PerMWCommitment(0.4))
env = BatchRequestParallelEnv(game, rounds=20, observation="public")
obs, info = env.reset(seed=0)
```

`examples/train_independent_pg.py` trains independent learners in this
environment. They reach the max-min fair point within 0.05% in about three
seconds on a laptop.

## Sequential batches

Real queues run batch after batch. `icgym.sequential.SequentialBatchGame` puts
a queue on a region over buses:
- applicants arrive at random buses with a size they need and a value per MW;
- each batch, the rule serves the queue on the capacity left by earlier
  batches, minus any load growth;
- served MW connect for good, and unserved need waits for the next batch.

`simulate(policy)` runs a whole queue and reports the MW connected, the MW
requested but not served, and the batches to connect. Its ground truth is the
batch-by-batch equilibrium: water-filling the residual region, capped at each
applicant's remaining need ([`myopic_equilibrium`](technical.md#myopic-batch-equilibrium)).
`icgym.sequential.SequentialBatchEnv` is the PettingZoo version, with agents
that arrive and leave. Agent slots never used by an arrival are not
terminated, which PettingZoo's API test reports as a warning.

## The operator as an agent

`icgym.rule_design.OperatorEnv` (Gymnasium) lets an operator choose the rule,
its minimum allocation, and the commitment for a batch on a sampled network.
Applicants answer at equilibrium ([`equilibrium_response`](technical.md#equilibrium-response)): the theory's
equilibrium where known, and best-response dynamics otherwise. The operator is
rewarded by an `Objective`: efficiency, minus waste and unfairness.
`SequentialOperatorEnv` does the same before every batch of a queue.

## Benchmark tasks

| Task | Answer |
|---|---|
| `uniform-ieee14` | unique equilibrium `w` (IEEE 14-bus, buses 9, 13, 14) |
| `uniform-random8` | unique equilibrium `w` (8 applicants, random 6-limit region) |
| `flatfee-ieee14` | the band `[w, w/beta]` |
| `proportional-limit` | requests inflated by `rho` (one limit, 4 applicants) |
| `location-activsg2000` | 3 pure equilibria over 6 ACTIVSg2000 buses |
| `generation-credit` | unique equilibrium, generators at the cap |
| `generation-gross-cycle` | no pure equilibrium |
| `nonfirm-two-tier` | firm and non-firm tiers served in full |

Three [baselines](technical.md#learning-baselines):
- **`best_response`:** best-response dynamics, an oracle that sees payoffs;
- **`hedge`:** full-information multiplicative weights on an action grid;
- **`independent_pg`:** independent policy gradient, which sees only its own
  rewards.

Each run is scored on:
- **[exploitability](technical.md#exploitability):** the largest gain from a unilateral deviation, over
  `V_i h_i`;
- **distance:** the gap to the predicted equilibrium, over `h_i`;
- **efficiency:** total served over the planner's optimum.

```bash
icgym-bench --list
icgym-bench --seeds 3 --json results.json          # the full suite, a few minutes
```

```python
from icgym.benchmarks import run, run_suite, format_results
print(format_results(run_suite(["uniform-ieee14"], ["independent_pg"], seeds=range(3))))
```

The results of the full suite are in `benchmarks/RESULTS.md` in the
repository. In summary:
- **Policy gradient** reaches the equilibrium wherever one exists
  (exploitability at most 0.0025).
- **No pure equilibrium:** every baseline stays exploitable, as the theory
  predicts.
- **Proportional rule with a minimum allocation:** best-response dynamics
  stall. Each best response there is a vanishing undercut of a rival, so the
  supremum is not attained.
