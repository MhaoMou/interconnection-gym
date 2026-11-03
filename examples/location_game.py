"""The ACTIVSg2000 location game: where do three applicants connect?

    python examples/location_game.py
"""

from icgym import activsg2000_location

game = activsg2000_location()               # shares of all 216 placements ship with the package
print("menu (buses):", game.menus[0])
print("pure equilibria:", game.pure_equilibria())
print("improving moves:", game.improving_moves())   # every one raises the sorted shares
prof, path, rounds = game.dynamics(("3010", "3010", "3010"))
print(f"best responses from three applicants at bus 3010 reach {prof} in {rounds} rounds;")
print("shares there (MW):", game.shares(prof).round(1))
