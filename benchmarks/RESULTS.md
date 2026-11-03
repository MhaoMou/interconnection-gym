# Benchmark results

All 8 tasks were run with all 3 baselines, 3 seeds each:

```bash
icgym-bench --seeds 3 --json benchmarks/results.json
```

Each table entry is computed over the final profile, averaged over seeds. The
columns are:
- **Exploitability:** the largest gain any applicant has from deviating
  alone, over its value at its hosting capacity.
- **Distance:** the largest gap to the predicted equilibrium, over the
  hosting capacity (served loads per tier).
- **Efficiency:** total served over the planner's optimum, or over the best
  total of any placement in the location game.
- **Converged:** the share of seeds in which best-response dynamics settled,
  meaning no request moved and no applicant could gain by deviating.

| task | baseline | exploitability (mean / max) | distance (mean) | efficiency (mean) | converged | seconds |
|---|---|---|---|---|---|---|
| uniform-ieee14 | best_response | 0.0000 / 0.0000 | 0.0000 | 0.807 | 3/3 | 0.3 |
| uniform-ieee14 | hedge | 0.0101 / 0.0101 | 0.0253 | 0.803 | - | 0.2 |
| uniform-ieee14 | independent_pg | 0.0002 / 0.0002 | 0.0006 | 0.807 | - | 0.1 |
| uniform-random8 | best_response | 0.0000 / 0.0000 | 0.0000 | 0.702 | 3/3 | 1.7 |
| uniform-random8 | hedge | 0.0125 / 0.0130 | 0.0355 | 0.699 | - | 1.6 |
| uniform-random8 | independent_pg | 0.0002 / 0.0002 | 0.0004 | 0.702 | - | 0.3 |
| flatfee-ieee14 | best_response | 0.0000 / 0.0000 | 0.0000 | 1.000 | 3/3 | 0.1 |
| flatfee-ieee14 | hedge | 0.0000 / 0.0000 | 0.0000 | 1.000 | - | 0.2 |
| flatfee-ieee14 | independent_pg | 0.0000 / 0.0000 | 0.0000 | 1.000 | - | 0.1 |
| proportional-limit | best_response | 0.3333 / 0.3333 | 0.0868 | 0.722 | 0/3 | 1.0 |
| proportional-limit | hedge | 0.0008 / 0.0008 | 0.0312 | 0.792 | - | 0.1 |
| proportional-limit | independent_pg | 0.0000 / 0.0000 | 0.0044 | 0.792 | - | 0.1 |
| location-activsg2000 | best_response | 0.0000 / 0.0000 | 0.0000 | 1.000 | 3/3 | 0.0 |
| location-activsg2000 | hedge | 0.0000 / 0.0000 | 0.0000 | 1.000 | - | 0.1 |
| location-activsg2000 | independent_pg | 0.0000 / 0.0000 | 0.0000 | 1.000 | - | 0.2 |
| generation-credit | best_response | 0.0000 / 0.0000 | 0.0000 | - | 3/3 | 0.6 |
| generation-credit | hedge | 0.0145 / 0.0145 | 0.0497 | - | - | 20.7 |
| generation-credit | independent_pg | 0.0014 / 0.0014 | 0.0080 | - | - | 5.1 |
| generation-gross-cycle | best_response | 0.0237 / 0.0259 | - | - | 0/3 | 4.7 |
| generation-gross-cycle | hedge | 0.0300 / 0.0300 | - | - | - | 17.9 |
| generation-gross-cycle | independent_pg | 0.0197 / 0.0205 | - | - | - | 4.6 |
| nonfirm-two-tier | best_response | 0.0000 / 0.0000 | 0.0000 | - | 3/3 | 1.2 |
| nonfirm-two-tier | hedge | 0.1421 / 0.1421 | 0.5204 | - | - | 32.5 |
| nonfirm-two-tier | independent_pg | 0.0011 / 0.0025 | 0.0038 | - | - | 8.9 |

## Reading

- **Policy gradient** reaches the theory's equilibrium wherever one exists.
  It uses only its own rewards. Its exploitability is at most 0.0025, and its
  distance is at most 0.8% of a hosting capacity.
- **Proportional rule with a minimum allocation:** best-response dynamics
  stall at profiles where all loadings are equal. There, each applicant's best
  response is a vanishing undercut of its rivals' loadings, which avoids the
  cut. The supremum is not attained, so the dynamics never settle (0/3
  converged). The learners find the inflated equilibrium.
- **Gross generation rule between the two margins:** every baseline stays
  exploitable, because no pure equilibrium exists.
- **Hedge** is limited by its action grid (41 points, or 17 × 5 for two-part
  strategies). Its error on the two-tier task comes from the coarse grid over
  the non-firm request.
