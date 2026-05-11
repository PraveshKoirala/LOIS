from __future__ import annotations

import argparse
from datetime import timedelta

from ortools.math_opt.python import mathopt

from cng import build_synthetic_cng
from lois1 import add_implies


def solve_bilevel_lois1(size: int, seed: int, time_limit_seconds: int = 100) -> dict[str, float]:
    cng = build_synthetic_cng(size, seed)
    cng.initialize()
    model = mathopt.Model(name="bilevel_lois1")

    defend = []
    attack = []
    for i in range(cng.size):
        defend.append(model.add_variable(lb=0, ub=1, is_integer=True, name=f"defend_{i}"))
        attack.append(model.add_variable(lb=0, ub=1, is_integer=True, name=f"attack_{i}"))

    defense_cost = sum(cng.cost_d[i] * defend[i] for i in range(cng.size))
    attack_cost = sum(cng.cost_a[i] * attack[i] for i in range(cng.size))
    model.add_linear_constraint(defense_cost <= cng.budget_d)
    model.add_linear_constraint(attack_cost <= cng.budget_a)

    threshold = 0
    counter = 0
    for i in range(cng.size):
        attacker_delta = cng.payoff_a[i] * (
            cng.gamma * (1 - defend[i]) + (1 - defend[i]) + (1 - cng.eta) * defend[i]
        )
        counter = add_implies(attacker_delta - threshold, 1 - attack[i] - 1, cng.budget_a - attack_cost - cng.cost_a[i], model, counter)
        counter = add_implies(-attacker_delta - threshold, attack[i] - 1, cng.budget_a - attack_cost + cng.cost_a[i], model, counter)

    payoff_d, _ = cng.get_payoffs(defend, attack)
    params = mathopt.SolveParameters(enable_output=False, time_limit=timedelta(seconds=time_limit_seconds))
    model.maximize(payoff_d)
    result = mathopt.solve(model, mathopt.SolverType.GSCIP, params=params)
    if result.termination.reason not in (mathopt.TerminationReason.OPTIMAL, mathopt.TerminationReason.FEASIBLE):
        raise RuntimeError(f"solve failed: {result.termination}")

    selected_defend = [int(round(result.variable_values()[defend[i]], 1)) for i in range(cng.size)]
    selected_attack = [int(round(result.variable_values()[attack[i]], 1)) for i in range(cng.size)]
    payoff_d_value, _ = cng.get_payoffs(selected_defend, selected_attack)
    return {
        "f_d": payoff_d_value,
        "PoS": cng.max_payoff_d / payoff_d_value,
        "is_lois1": float(cng.is_lois1(selected_defend, selected_attack)),
        "tl": float(result.termination.reason == mathopt.TerminationReason.FEASIBLE),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve the bilevel LOIS-1 critical node game instance.")
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--time-limit", type=int, default=100)
    args = parser.parse_args()

    result = solve_bilevel_lois1(args.size, args.seed, args.time_limit)
    for key in ("f_d", "PoS", "is_lois1", "tl"):
        print(f"{key}: {result[key]}")


if __name__ == "__main__":
    main()
