"""Check that PGO evidence belongs to the measured Ledger hot path.

Auxiliary executables and test files are intentionally absent from benchmark
training. Missing data for those files is recorded; missing or stale data for
the four trained translation units rejects the build.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess
from typing import Any


CORE_SOURCES = ("book.cpp", "reference_book.cpp", "feed.cpp", "bench_replay.cpp")
REQUIRED_LLVM_FUNCTIONS = {
    "FastBook::process": "_ZN6ledger8FastBook7processERKNS_7CommandERNS_11EventBufferE",
    "ReferenceBook::process": "_ZN6ledger13ReferenceBook7processERKNS_7CommandERNS_11EventBufferE",
}
_PROFILE_DIAGNOSTIC = re.compile(
    r"(?:warning:|error:).*?(?:"
    r"-W(?:error=)?(?:profile-instr-(?:unprofiled|out-of-date)|missing-profile|coverage-mismatch)"
    r"|profile (?:count )?data.*(?:not found|mismatch|out of date|ignored)"
    r"|no profile data available)", re.IGNORECASE
)
_SOURCE = re.compile(r"(?<![\w.])([A-Za-z_][\w.-]*\.cpp)(?![\w])")


def profile_diagnostics(text: str) -> list[dict[str, Any]]:
    """Associate Ninja/Make's buffered diagnostics with their translation unit."""
    target: str | None = None
    found: list[dict[str, Any]] = []
    for line in text.splitlines():
        if "Building CXX object " in line:
            names = _SOURCE.findall(line)
            target = names[-1] if names else None
        if not _PROFILE_DIAGNOSTIC.search(line):
            continue
        names = _SOURCE.findall(line)
        source = names[-1] if names else target
        found.append({"source": source, "core": source in CORE_SOURCES,
                      "unassigned": source is None, "message": line.strip()[:2000]})
    return found


def _profile_path(build_dir: Path, flags: list[str], prefix: str) -> Path:
    selected = [flag[len(prefix):] for flag in flags if flag.startswith(prefix)]
    if len(selected) != 1 or not selected[0]:
        raise ValueError(f"Expected exactly one {prefix} profile-use flag")
    path = Path(selected[0]).resolve()
    try:
        path.relative_to((build_dir / "profile").resolve())
    except ValueError as exc:
        raise ValueError("Profile-use path must stay inside this build's profile directory") from exc
    return path


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _llvm_counts(profile: Path) -> dict[str, int]:
    # Tool and arguments are fixed; model text is never an executable or command.
    completed = subprocess.run(
        ["llvm-profdata", "show", "--all-functions", "--counts", str(profile)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        check=False,
    )
    if completed.returncode:
        raise ValueError("llvm-profdata could not read the merged profile: " + completed.stderr[-2000:])
    counts: dict[str, int] = {}
    for name, symbol in REQUIRED_LLVM_FUNCTIONS.items():
        match = re.search(r"^  " + re.escape(symbol) +
                          r":\s*\n(?:(?!^  \S+:).)*?Function count:\s*(\d+)",
                          completed.stdout, re.MULTILINE | re.DOTALL)
        if match:
            counts[name] = int(match.group(1))
    if any(counts.get(name, 0) <= 0 for name in REQUIRED_LLVM_FUNCTIONS):
        raise ValueError("Merged profile has no positive execution counts for Ledger's core process functions")
    return counts


def verify_profiles(compiler: str, build_dir: Path, phase_flags: list[str]) -> dict[str, Any]:
    """Fail closed on absent, stale or unreadable profiles for trained Ledger code.

    This verifies artifact and compiler diagnostic evidence, not a performance
    benefit. Ordinary non-PGO builds should not call this function.
    """
    result: dict[str, Any] = {"correctness": False, "reason": None,
                              "profile_hashes": {}, "counts": {}, "diagnostics": []}
    try:
        if compiler not in ("gcc", "clang"):
            raise ValueError("compiler must be gcc or clang")
        if not isinstance(phase_flags, list) or not all(isinstance(flag, str) for flag in phase_flags):
            raise ValueError("phase_flags must be a list of strings")
        directory = Path(build_dir).resolve()
        log = directory / "build.log"
        if not log.is_file():
            raise ValueError("Final profile-use build log is missing")
        diagnostics = profile_diagnostics(log.read_text(errors="replace"))
        result["diagnostics"] = diagnostics
        rejected = [item for item in diagnostics if item["core"] or item["unassigned"]]
        if rejected:
            raise ValueError("Profile data was missing or ignored for trained core code: " +
                             "; ".join(item["message"] for item in rejected)[:2000])
        if compiler == "clang":
            if any(flag.startswith("-fprofile-instr-generate") for flag in phase_flags):
                raise ValueError("Final Clang flags still enable profile generation")
            profile = _profile_path(directory, phase_flags, "-fprofile-instr-use=")
            if not profile.is_file() or not profile.stat().st_size:
                raise ValueError("Merged Clang profile is missing or empty")
            result["profile_hashes"][str(profile.relative_to(directory))] = _hash(profile)
            result["counts"] = _llvm_counts(profile)
        else:
            if any(flag.startswith("-fprofile-generate") for flag in phase_flags):
                raise ValueError("Final GCC flags still enable profile generation")
            profile = _profile_path(directory, phase_flags, "-fprofile-use=")
            if not profile.is_dir():
                raise ValueError("GCC profile directory is missing")
            files = list(profile.rglob("*.gcda"))
            for source in CORE_SOURCES:
                matches = [path for path in files
                           if path.name == source + ".gcda" or path.name.endswith("#" + source + ".gcda")]
                if not matches:
                    raise ValueError(f"GCC training produced no core profile for {source}")
                for path in matches:
                    with path.open("rb") as stream:
                        header = stream.read(16)
                    if len(header) < 16 or header[:4] not in (b"adcg", b"gcda") or path.stat().st_size <= 16:
                        raise ValueError(f"GCC core profile is corrupt or empty: {source}")
                    result["profile_hashes"][str(path.relative_to(directory))] = _hash(path)
        result["correctness"] = True
        result["reason"] = "Core profile artifacts are present and the compiler reported no missing or stale core data"
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result["reason"] = str(exc)[-4000:]
    return result
