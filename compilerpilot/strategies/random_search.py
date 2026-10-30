"""Uniform random search over the allow-listed compiler space."""
from __future__ import annotations

import random
from typing import Any, Iterable

from compilerpilot.space import Candidate, sample_candidate


class RandomSearch:
    """Seeded sampling without repeated candidates within an experiment."""

    def __init__(self, seed: int = 0, compilers: Iterable[str] = ("gcc", "clang")):
        self.rng = random.Random(seed)
        self.compilers = tuple(compilers)
        if not self.compilers or any(c not in ("gcc", "clang") for c in self.compilers):
            raise ValueError("compilers must contain gcc and/or clang")
        self._proposed: set[str] = set()

    def propose(self, history: list[dict[str, Any]], seen: set[str]) -> Candidate | None:
        """Return an unseen candidate, or None when sampling finds no free point.

        Internal tracking also prevents repeats when a caller rejects a proposal
        before recording a trial. Failed trials in ``seen`` still consume a build.
        """
        excluded = set(seen) | self._proposed
        for row in history:
            try:
                excluded.add(Candidate.from_dict(row["candidate"]).hash())
            except (KeyError, TypeError, ValueError):
                continue
        for _ in range(10000):
            candidate = sample_candidate(self.rng, self.compilers)
            if candidate.hash() not in excluded:
                self._proposed.add(candidate.hash())
                return candidate
        return None
