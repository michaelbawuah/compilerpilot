#!/usr/bin/env python3
"""Rebuild the report from recorded results, with no benchmark side effects."""
import argparse
import json
from pathlib import Path

from compilerpilot.plot import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--expected-seeds", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(generate_report(args.results, budget=args.budget, expected_seeds=args.expected_seeds), indent=2))


if __name__ == "__main__":
    main()
