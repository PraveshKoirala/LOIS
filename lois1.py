from __future__ import annotations

import argparse
from datetime import timedelta

from ortools.math_opt.python import mathopt

from cng import build_synthetic_cng


BIG_M = 99999


def add_implies(p, q, r, model: mathopt.Model, counter: int) -> int:
    m1 = model.add_variable(lb=0, ub=1, is_integer=True, name=f"m{counter}_1")
    m2 = model.add_variable(lb=0, ub=1, is_integer=True, name=f"m{counter}_2")
    m3 = model.add_variable(lb=0, ub=1, is_integer=True, name=f"m{counter}_3")
    model.add_linear_constraint(m1 + m2 + m3 >= 1)
    epsilon = 1e-5
    model.add_linear_constraint(p - BIG_M * (1 - m1) <= 0)
    model.add_linear_constraint(q - BIG_M * (1 - m2) <= -epsilon)
    model.add_linear_constraint(r - BIG_M * (1 - m3) <= -epsilon)
    return counter + 1


def solve_lois1(size: int, seed: int, time_limit_seconds: int = 100) -> dict[str, float]:
    cng = build_synthetic_cng(size, seed)
    cng.initialize()
    model = mathopt.Model(name="lois1")

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
        defender_delta = cng.payoff_d[i] * (
            -(1 - attack[i]) + cng.eta * attack[i] + cng.epsilon * (1 - attack[i]) - cng.delta * attack[i]
        )
        counter = add_implies(defender_delta - threshold, 1 - defend[i] - 1, cng.budget_d - defense_cost - cng.cost_d[i], model, counter)
        counter = add_implies(-defender_delta - threshold, defend[i] - 1, cng.budget_d - defense_cost + cng.cost_d[i], model, counter)

    for i in range(cng.size):
        attacker_delta = cng.payoff_a[i] * (
            cng.gamma * (1 - defend[i]) + (1 - defend[i]) + (1 - cng.eta) * defend[i]
        )
        counter = add_implies(attacker_delta - threshold, 1 - attack[i] - 1, cng.budget_a - attack_cost - cng.cost_a[i], model, counter)
        counter = add_implies(-attacker_delta - threshold, attack[i] - 1, cng.budget_a - attack_cost + cng.cost_a[i], model, counter)

    payoff_d, payoff_a = cng.get_payoffs(defend, attack)
    params = mathopt.SolveParameters(enable_output=False, time_limit=timedelta(seconds=time_limit_seconds))

    model.maximize(payoff_a)
    attacker_result = mathopt.solve(model, mathopt.SolverType.GSCIP, params=params)
    if attacker_result.termination.reason not in (mathopt.TerminationReason.OPTIMAL, mathopt.TerminationReason.FEASIBLE):
        raise RuntimeError(f"attacker solve failed: {attacker_result.termination}")
    attacker_defend = [int(round(attacker_result.variable_values()[defend[i]], 1)) for i in range(cng.size)]
    attacker_attack = [int(round(attacker_result.variable_values()[attack[i]], 1)) for i in range(cng.size)]
    _, attacker_payoff_a = cng.get_payoffs(attacker_defend, attacker_attack)

    model.maximize(payoff_d)
    defender_result = mathopt.solve(model, mathopt.SolverType.GSCIP, params=params)
    if defender_result.termination.reason not in (mathopt.TerminationReason.OPTIMAL, mathopt.TerminationReason.FEASIBLE):
        raise RuntimeError(f"defender solve failed: {defender_result.termination}")
    final_defend = [int(round(defender_result.variable_values()[defend[i]], 1)) for i in range(cng.size)]
    final_attack = [int(round(defender_result.variable_values()[attack[i]], 1)) for i in range(cng.size)]
    final_payoff_d, _ = cng.get_payoffs(final_defend, final_attack)

    return {
        "f_a": attacker_payoff_a,
        "PoA": cng.max_payoff_a / attacker_payoff_a,
        "attacker_is_lois1": float(cng.is_lois1(attacker_defend, attacker_attack)),
        "f_d": final_payoff_d,
        "PoD": cng.max_payoff_d / final_payoff_d,
        "defender_is_lois1": float(cng.is_lois1(final_defend, final_attack)),
        "tl": float(
            attacker_result.termination.reason == mathopt.TerminationReason.FEASIBLE
            or defender_result.termination.reason == mathopt.TerminationReason.FEASIBLE
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve the LOIS-1 critical node game instance.")
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--time-limit", type=int, default=100)
    args = parser.parse_args()

    result = solve_lois1(args.size, args.seed, args.time_limit)
    for key in ("f_a", "PoA", "tl", "f_d", "PoD", "attacker_is_lois1", "defender_is_lois1"):
        print(f"{key}: {result[key]}")


if __name__ == "__main__":
    main()
