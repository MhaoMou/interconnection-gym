"""Quick start: a batch region, the uniform rule, and the theory's answer.

    python examples/quickstart.py
"""

import numpy as np

from icgym import BatchGame, PerMWCommitment, Region, UniformRule, predicted_equilibrium

# Three applicants, two shared limits: s1 + s2 <= 2 and s2 + s3 <= 3 (MW).
region = Region(A=[[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], b=[2.0, 3.0])
game = BatchGame(region, UniformRule(beta=0.5), PerMWCommitment(kappa=0.4), value=1.0)

w = predicted_equilibrium(game)["requests"]          # the max-min fair point, here [1, 1, 2]
print("equilibrium requests:", w)
print("served at the equilibrium:", game.allocate(w))
print("payoffs:", game.payoffs(w))

inflated = np.array([1.5, 1.0, 2.0])                 # applicant 0 over-requests
print("gain from deviating at", inflated, ":", game.exploitability(inflated).round(4))
