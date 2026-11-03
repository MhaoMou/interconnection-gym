# interconnection-gym

A simulator, a set of reinforcement-learning environments, and a benchmark
for the **batch interconnection of large loads** (data centers and similar
loads). Applicants request grid capacity, the operator splits what the network
can host among them with an allocation rule, and every applicant pays for what
it requested.

- **Operators and analysts** compare allocation rules and fees on their own
  network before adopting them.
- **Researchers** get multi-agent games whose equilibria are known in closed
  form, so learned behavior can be scored against ground truth.

## Install

```bash
pip install interconnection-gym            # core: numpy and scipy only
pip install "interconnection-gym[envs]"    # adds the Gymnasium and PettingZoo environments
```

## Quick start

```python
from icgym import BatchGame, PerMWCommitment, Region, UniformRule, predicted_equilibrium

region = Region(A=[[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], b=[2.0, 3.0])   # two shared limits
game = BatchGame(region, UniformRule(beta=0.5), PerMWCommitment(kappa=0.4), value=1.0)

w = predicted_equilibrium(game)["requests"]   # the unique equilibrium: [1, 1, 2]
game.allocate(w), game.payoffs(w)
game.exploitability([1.5, 1.0, 2.0])          # applicant 0 gains 0.2 by returning to w
```

**Compare rules on a batch** (operators):

```bash
icgym-compare --data path/to/scenario --buses 9,13,14 --kappa 0.4 --beta 0.5 --html report.html
```

**Train agents and score them** (researchers):

```python
from icgym.envs import BatchRequestParallelEnv       # PettingZoo ParallelEnv
from icgym.benchmarks import run

env = BatchRequestParallelEnv(game, rounds=20, observation="public")
run("uniform-ieee14", "independent_pg", seed=0)      # exploitability, distance to the equilibrium, efficiency
```

## What is in it

- **Core:**
  - batch regions, from a network study or synthetic;
  - the uniform, proportional, and utilitarian rules, with a minimum
    allocation;
  - per-MW, flat, and deposit fees;
  - the batch request game.
- **Theory oracles:** the max-min fair equilibrium, the flat-fee band, the
  proportional rule's request inflation, the price-of-anarchy bound, and the
  location, generation, and non-firm results of the accompanying paper.
- **Extensions:** location choice, on-site generation and storage (gross or
  crediting rule), and firm and non-firm service.
- **Networks:**
  - an exact exporter of batch regions from DC power-flow scenarios;
  - `DCOPFRegion` for batches of any size on large networks.
- **Data:**
  - bundled regions of the synthetic ACTIVSg2000 Texas grid;
  - IEEE 14-bus regions, built locally from the public case file on first
    use.

  See [docs/data.md](docs/data.md) for provenance and licenses.
- **Environments and benchmarks:**
  - PettingZoo and Gymnasium environments, and a vectorized environment;
  - eight benchmark tasks with known answers;
  - three baselines, with results in
    [benchmarks/RESULTS.md](benchmarks/RESULTS.md).

## Documentation

The `docs/` folder (built with `mkdocs build`) has these pages:
- [model and theory](docs/theory.md);
- [comparing rules](docs/operators.md);
- [regions from a network](docs/networks.md);
- [environments and benchmarks](docs/researchers.md);
- [data and licenses](docs/data.md);
- [development](docs/development.md).

Runnable examples are in [examples/](examples/).

## License

The code is MIT licensed. The bundled ACTIVSg2000-derived data is an
adaptation of a CC BY 3.0 work. The IEEE 14-bus case is not redistributed:
it is downloaded from its source on first use. See
[docs/data.md](docs/data.md).
