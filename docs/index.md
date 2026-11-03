# interconnection-gym

A simulator, a set of reinforcement-learning environments, and a benchmark
for the **batch interconnection of large loads** (data centers and similar
loads). Applicants request grid capacity, the operator splits what the network
can host among them with an allocation rule, and every applicant pays for what
it requested.

- **Operators and analysts** can compare allocation rules and fees on their
  own network before adopting them: [Comparing rules](operators.md).
- **Researchers** get multi-agent games whose equilibria are known in closed
  form, so learned behavior is scored against ground truth:
  [Environments and benchmarks](researchers.md).

## Install

```bash
# not yet on PyPI: install from GitHub
pip install "interconnection-gym @ git+https://github.com/MhaoMou/interconnection-gym"          # core: numpy and scipy
pip install "interconnection-gym[envs] @ git+https://github.com/MhaoMou/interconnection-gym"    # adds Gymnasium and PettingZoo
```

Python 3.10 or newer.

## Thirty seconds

```python
from icgym import BatchGame, PerMWCommitment, Region, UniformRule, predicted_equilibrium

region = Region(A=[[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], b=[2.0, 3.0])   # two shared limits
game = BatchGame(region, UniformRule(beta=0.5), PerMWCommitment(kappa=0.4), value=1.0)

w = predicted_equilibrium(game)["requests"]   # the unique equilibrium: [1, 1, 2]
game.allocate(w), game.payoffs(w)
game.exploitability([1.5, 1.0, 2.0])          # applicant 0 gains 0.2 by returning to w
```

## Contents

| Page | What it covers |
|---|---|
| [Model and theory](theory.md) | What is modeled, and the results the oracles return |
| [Comparing rules](operators.md) | `icgym-compare` and `compare_rules` |
| [Regions from a network](networks.md) | Exporting a batch region from a DC power-flow study; the implicit `DCOPFRegion` |
| [Environments and benchmarks](researchers.md) | PettingZoo and Gymnasium environments, the vectorized environment, benchmark tasks and baselines |
| [Data and licenses](data.md) | The bundled IEEE 14-bus and ACTIVSg2000 catalogs, where they come from, and their terms |
| [Development](development.md) | Tests, lint, and the release checklist |
