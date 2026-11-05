# Comparing rules

`icgym-compare` finds each rule's equilibrium on one batch. Where the theory
gives the answer, it uses that and checks it numerically. Elsewhere it runs
[best-response dynamics](technical.md#best-response-dynamics) from the max-min fair point.

For each rule it reports:
- served MW and the planner's optimum;
- efficiency and the price of anarchy, with its bound for the uniform rule;
- request inflation and the commitment paid on unserved MW;
- the binding limits;
- exploitability (zero at an equilibrium).

```bash
# a region file, e.g. from icgym-export
icgym-compare --region region.json --kappa 0.4 --beta 0.5 --html report.html

# straight from a DC power-flow study (exports the region first)
icgym-compare --data path/to/scenario --buses 9,13,14 --rules uniform,proportional

# any batch size, exact network region; only closed-form equilibria unless --slow
icgym-compare --data path/to/activsg2000 --buses 7106,5430,5276,7259,5250 --implicit
```

```python
from icgym import PerMWCommitment, ieee14
from icgym.compare import compare_rules, format_table, report_html

reports = compare_rules(ieee14((9, 13, 14)), ["uniform", "proportional", "utilitarian"],
                        fee=PerMWCommitment(1 / 2.46), beta=0.0)
print(format_table(reports))
report_html(reports, "report.html")
```

## Reading the report

| Column | Meaning |
|---|---|
| method | `theory` (closed form), `dynamics` (best responses settled), `no convergence`, or `skipped` (implicit region without `--slow`) |
| served MW / planner MW | total served at the equilibrium, and the largest total the region can host |
| efficiency, PoA | their ratio, and its inverse |
| PoA bound | the theory's bound for the uniform rule |
| max inflation | the largest request over its served load (1 means every request is honest) |
| unserved request MW | MW requested and paid for but not served |
| exploitability | the largest gain any applicant has from changing its request alone |

## Example

IEEE 14-bus, buses 9, 13 and 14, with value over commitment of 2.46 and no
minimum allocation. These numbers reproduce the paper's comparison:

| Rule | Served (MW) | Efficiency | Max inflation |
|---|---|---|---|
| Uniform | 46.8, 25.4, 25.4 | 0.807 | 1.000 |
| Proportional | 54.9, 20.3, 29.7 | 0.867 | 1.686 |
| Utilitarian | 91.5, 14.4, 15.0 | 1.000 | 1.000 |

How to read the three rows:
- **Uniform rule:** it serves less in total than the planner could, but
  every request is honest.
- **Proportional rule:** it inflates every request by 1.69, so applicants pay
  for capacity they are not served.
- **Utilitarian rule:** at this batch's equilibrium it reaches the planner's
  optimum, but it concentrates service on the site that loads the limits
  least.

`examples/compare_rules.py` runs this comparison.
