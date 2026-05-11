from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
import random


def _scaled_int(value: float | Fraction) -> int:
    return int(Fraction(value) * 100)


def solve_knapsack(capacity: float, weights: list[int], values: list[int]) -> int:
    scaled_capacity = _scaled_int(capacity)
    scaled_weights = [_scaled_int(weight) for weight in weights]
    dp = [0] * (scaled_capacity + 1)
    for weight, value in zip(scaled_weights, values):
        for remaining in range(scaled_capacity, weight - 1, -1):
            dp[remaining] = max(dp[remaining], dp[remaining - weight] + value)
    return max(dp)


@dataclass
class CNG:
    size: int
    cost_a: list[int]
    cost_d: list[int]
    budget_a: float
    budget_d: float
    payoff_a: list[int]
    payoff_d: list[int]
    gamma: float
    eta: float
    epsilon: float
    delta: float

    def initialize(self) -> None:
        self.max_payoff_d = sum(self.payoff_d)
        self.max_payoff_a = solve_knapsack(self.budget_a, self.cost_a, self.payoff_a)

    def get_payoffs(self, defend: list[int], attack: list[int]) -> tuple[float, float]:
        attacker = 0.0
        defender = 0.0
        for i in range(self.size):
            attacker += self.payoff_a[i] * (
                -self.gamma * (1 - defend[i]) * (1 - attack[i])
                + (1 - defend[i]) * attack[i]
                + (1 - self.eta) * defend[i] * attack[i]
            )
            defender += self.payoff_d[i] * (
                (1 - defend[i]) * (1 - attack[i])
                + self.eta * defend[i] * attack[i]
                + self.epsilon * defend[i] * (1 - attack[i])
                + self.delta * (1 - defend[i]) * attack[i]
            )
        return defender, attacker

    def get_costs(self, defend: list[int], attack: list[int]) -> tuple[int, int]:
        return (
            sum(defend[i] * self.cost_d[i] for i in range(self.size)),
            sum(attack[i] * self.cost_a[i] for i in range(self.size)),
        )

    def is_feasible(self, defend: list[int], attack: list[int]) -> bool:
        if not all(value in (0, 1) for value in defend + attack):
            return False
        cost_d, cost_a = self.get_costs(defend, attack)
        return cost_d <= self.budget_d and cost_a <= self.budget_a

    def is_lois1(self, defend: list[int], attack: list[int]) -> bool:
        baseline_d, baseline_a = self.get_payoffs(defend, attack)
        for i in range(self.size):
            flipped_defend = defend[:]
            flipped_defend[i] = 1 - flipped_defend[i]
            new_d, _ = self.get_payoffs(flipped_defend, attack)
            if self.is_feasible(flipped_defend, attack) and new_d > baseline_d + 1e-5:
                return False
        for i in range(self.size):
            flipped_attack = attack[:]
            flipped_attack[i] = 1 - flipped_attack[i]
            _, new_a = self.get_payoffs(defend, flipped_attack)
            if self.is_feasible(defend, flipped_attack) and new_a > baseline_a + 1e-5:
                return False
        return True

    def is_lois2(self, defend: list[int], attack: list[int]) -> bool:
        baseline_d, baseline_a = self.get_payoffs(defend, attack)
        for i, j in combinations(range(self.size), 2):
            flipped_defend = defend[:]
            flipped_defend[i] = 1 - flipped_defend[i]
            flipped_defend[j] = 1 - flipped_defend[j]
            new_d, _ = self.get_payoffs(flipped_defend, attack)
            if self.is_feasible(flipped_defend, attack) and new_d > baseline_d + 1e-5:
                return False
        for i, j in combinations(range(self.size), 2):
            flipped_attack = attack[:]
            flipped_attack[i] = 1 - flipped_attack[i]
            flipped_attack[j] = 1 - flipped_attack[j]
            _, new_a = self.get_payoffs(defend, flipped_attack)
            if self.is_feasible(defend, flipped_attack) and new_a > baseline_a + 1e-5:
                return False
        return True


def build_synthetic_cng(size: int, seed: int) -> CNG:
    random.seed(seed)
    upper_cost = 25
    cost_a = [random.randint(1, upper_cost) for _ in range(size)]
    cost_d = [random.randint(1, upper_cost) for _ in range(size)]
    budget_a = random.choice([0.03, 0.1, 0.3]) * sum(cost_a)
    budget_d = random.choice([0.3, 0.75]) * sum(cost_d)
    payoff_a = [cost_a[i] + random.randint(1, upper_cost) for i in range(size)]
    payoff_d = [cost_d[i] + random.randint(1, upper_cost) for i in range(size)]
    gamma = random.choice([0, 0.1])
    eta = random.choice([0.6, 0.8])
    epsilon = 1.25 * eta
    delta = 0.8 * eta
    return CNG(
        size=size,
        cost_a=cost_a,
        cost_d=cost_d,
        budget_a=budget_a,
        budget_d=budget_d,
        payoff_a=payoff_a,
        payoff_d=payoff_d,
        gamma=gamma,
        eta=eta,
        epsilon=epsilon,
        delta=delta,
    )
