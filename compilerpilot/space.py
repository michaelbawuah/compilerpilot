"""The finite compiler search space. No flag or executable comes from model text."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any, Literal

Compiler = Literal["gcc", "clang"]
KNOBS: dict[str, list[Any]] = {
    "compiler": ["gcc", "clang"],
    "opt": ["O2", "O3"],
    "isa": ["none", "x86-64-v3", "native"],
    "lto": [False, True],
    "pgo": [False, True],
    "inline": ["default", 100, 500],
    "unroll": [False, True],
    "vectorize": [True, False],
    "frame_pointer": [False, True],
    "alignment": [0, 32, 64],
    "semantic_interposition": [True, False],
}


@dataclass(frozen=True)
class Candidate:
    compiler: Compiler = "gcc"
    opt: str = "O3"
    isa: str = "native"
    lto: bool = False
    pgo: bool = False
    inline: str | int = "default"
    unroll: bool = False
    vectorize: bool = True
    frame_pointer: bool = False
    alignment: int = 0
    semantic_interposition: bool = True

    def __post_init__(self) -> None:
        for name, allowed in KNOBS.items():
            value = getattr(self, name)
            if not any(type(value) is type(item) and value == item for item in allowed):
                raise ValueError(f"Invalid {name}: {value!r}; allowed values: {allowed!r}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Candidate":
        if not isinstance(data, dict):
            raise ValueError("A candidate must be a JSON object")
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown candidate knobs: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()[:16]


def compiler_executable(candidate: Candidate) -> str:
    return {"gcc": "g++", "clang": "clang++"}[candidate.compiler]


def list_knobs() -> dict[str, Any]:
    """Return independent copies so tool callers cannot modify the allow-list."""
    return {name: list(values) for name, values in KNOBS.items()}


def sample_candidate(rng: random.Random, compilers: tuple[str, ...] = ("gcc", "clang")) -> Candidate:
    if not compilers or any(c not in KNOBS["compiler"] for c in compilers):
        raise ValueError("compilers must contain gcc and/or clang")
    data = {name: rng.choice(values) for name, values in KNOBS.items()}
    data["compiler"] = rng.choice(compilers)
    return Candidate.from_dict(data)


def to_flags(candidate: Candidate, phase: str = "normal", profile_dir: str | Path | None = None) -> list[str]:
    """Translate validated values to argv elements, including two-stage PGO flags.

    Paths are harness-owned and remain arguments; callers must use shell=False.
    Clang profile-use expects profile_dir/merged.profdata after llvm-profdata merge.
    GCC must generate and consume profiles in the same object directory.
    """
    # Revalidate even if a caller bypassed frozen-dataclass initialization.
    Candidate.from_dict(candidate.to_dict())
    if phase not in ("normal", "generate", "use"):
        raise ValueError(f"Unknown PGO phase: {phase}")
    if phase != "normal" and not candidate.pgo:
        raise ValueError("PGO phase requires candidate.pgo=True")
    if candidate.pgo and (phase == "normal" or profile_dir is None):
        raise ValueError("PGO requires a generate/use phase and a profile directory")
    flags = [f"-{candidate.opt}", "-DNDEBUG"]
    if candidate.isa != "none":
        flags.append(f"-march={candidate.isa}")
    if candidate.lto:
        flags.extend(["-flto"] if candidate.compiler == "gcc" else ["-flto=thin", "-fuse-ld=lld"])
    if candidate.inline != "default":
        if candidate.compiler == "gcc":
            flags.append(f"-finline-limit={candidate.inline}")
        else:
            flags.extend(["-mllvm", f"-inline-threshold={candidate.inline}"])
    if candidate.unroll:
        flags.append("-funroll-loops")
    if not candidate.vectorize:
        flags.extend(["-fno-tree-vectorize"] if candidate.compiler == "gcc" else ["-fno-vectorize", "-fno-slp-vectorize"])
    if candidate.frame_pointer:
        flags.append("-fno-omit-frame-pointer")
    if candidate.alignment:
        flags.append(f"-falign-functions={candidate.alignment}")
    if not candidate.semantic_interposition:
        flags.append("-fno-semantic-interposition")
    if candidate.pgo:
        profile = str(Path(profile_dir).resolve())
        if any(ch in profile for ch in ("\x00", "\n", "\r", ";", "`", "$")):
            raise ValueError("Invalid profile directory")
        if candidate.compiler == "gcc":
            flags.extend([f"-fprofile-{phase}={profile}", "-fprofile-correction"] if phase == "use" else [f"-fprofile-generate={profile}"])
        elif phase == "generate":
            flags.append(f"-fprofile-instr-generate={profile}/default-%p.profraw")
        else:
            flags.append(f"-fprofile-instr-use={profile}/merged.profdata")
    return flags
