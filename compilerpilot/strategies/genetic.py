"""Tournament-selection genetic search over compiler knob chromosomes."""
from __future__ import annotations

import math
from typing import Any, Iterable

from compilerpilot.space import Candidate, sample_candidate
from compilerpilot.strategies.random_search import RandomSearch


def _fitness(row: dict[str, Any]) -> float | None:
    if row.get("correctness") is not True or row.get("measurement_valid", True) is not True:
        return None
    value = row.get("throughput")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 and math.isfinite(value) else None


class GeneticSearch(RandomSearch):
    """Steady-state population (8), tournament selection, and per-knob mutation.

    Each generation is populated from the highest-throughput correct measured
    trials. A single cut mixes the two parent chromosomes. The caller evaluates
    one child at a time, so compile/measurement costs stay comparable to random.
    """

    def __init__(self, seed: int = 0, compilers: Iterable[str] = ("gcc", "clang"),
                 population: int = 8, mutation_rate: float = 0.2):
        super().__init__(seed, compilers)
        if population < 2:
            raise ValueError("population must be at least two")
        if not 0 <= mutation_rate <= 1:
            raise ValueError("mutation_rate must be between zero and one")
        self.population = population
        self.mutation_rate = mutation_rate

    def propose(self, history: list[dict[str, Any]], seen: set[str]) -> Candidate | None:
        valid: dict[str, tuple[Candidate, float]] = {}
        for row in history:
            score = _fitness(row)
            if score is None:
                continue
            try:
                candidate = Candidate.from_dict(row["candidate"])
            except (KeyError, TypeError, ValueError):
                continue
            if candidate.compiler not in self.compilers:
                continue
            valid[candidate.hash()] = (candidate, score)
        if len(valid) < self.population:
            return super().propose(history, seen)
        population = sorted(valid.values(), key=lambda item: (-item[1], item[0].hash()))[:self.population]
        excluded = set(seen) | self._proposed
        for row in history:
            try:
                excluded.add(Candidate.from_dict(row["candidate"]).hash())
            except (KeyError, TypeError, ValueError):
                continue

        def tournament() -> Candidate:
            contestants = self.rng.sample(population, min(3, len(population)))
            return max(contestants, key=lambda item: item[1])[0]

        for _ in range(200):
            left, right = tournament().to_dict(), tournament().to_dict()
            keys = list(left)
            cut = self.rng.randrange(1, len(keys))
            child = {key: (left if i < cut else right)[key] for i, key in enumerate(keys)}
            mutation = sample_candidate(self.rng, self.compilers).to_dict()
            for key in keys:
                if self.rng.random() < self.mutation_rate:
                    child[key] = mutation[key]
            candidate = Candidate.from_dict(child)
            if candidate.hash() not in excluded:
                self._proposed.add(candidate.hash())
                return candidate
        return super().propose(history, seen)
