"""Build validated candidates without modifying Ledger or invoking a shell."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import time
from typing import Any

from .space import Candidate, compiler_executable, to_flags


@dataclass
class CommandResult:
    argv: list[str]
    returncode: int
    elapsed_s: float
    output: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def run_command(argv: list[str | Path], *, timeout: float, log: Path,
                cwd: Path | None = None) -> CommandResult:
    """Keep full output on disk and return only the last 12 KiB.

    A timed-out compiler can have children; kill its process group as well.
    Arguments are always passed directly to exec, never through a shell.
    """
    args = [str(arg) for arg in argv]
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    timed_out = False
    with log.open("wb") as output:
        try:
            process = subprocess.Popen(args, cwd=cwd, stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        except OSError as exc:
            output.write(str(exc).encode())
            return CommandResult(args, 127, time.perf_counter() - start, str(exc))
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            code = process.wait()
    with log.open("rb") as output:
        output.seek(max(0, log.stat().st_size - 12288))
        tail = output.read().decode(errors="replace")
    if timed_out:
        tail = f"Timed out after {timeout}s\n" + tail
    return CommandResult(args, code, time.perf_counter() - start, tail, timed_out)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_provenance(ledger: Path) -> dict[str, Any]:
    """Record both git identity and bytes, so scratch mutation tests are identifiable."""
    digest = hashlib.sha256()
    for path in sorted(ledger.rglob("*")):
        relative = path.relative_to(ledger)
        if path.is_file() and (relative.parts[0] in {"src", "include", "bench", "tests"}
                               or path.name == "CMakeLists.txt"):
            if "results" in relative.parts or "__pycache__" in relative.parts:
                continue
            digest.update(str(relative).encode() + b"\0")
            digest.update(bytes.fromhex(file_sha256(path)))
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", "-C", str(ledger), *args],
                                           stderr=subprocess.DEVNULL, timeout=10).decode().strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"ledger_commit": git("rev-parse", "HEAD"),
            "ledger_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "source_sha256": digest.hexdigest()}


@dataclass
class BuildResult:
    build_id: str
    candidate: dict[str, Any]
    build_dir: str
    success: bool = False
    compile_s: float = 0.0
    text_bytes: int | None = None
    compiler_version: str = ""
    error: str | None = None
    flags: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BuildResult":
        return cls(**data)


class BuildManager:
    def __init__(self, ledger: Path, root: Path, build_timeout: float = 300,
                 configure_timeout: float = 120, jobs: int = 2):
        self.ledger = Path(ledger).resolve()
        self.root = Path(root).resolve()
        self.build_timeout = build_timeout
        self.configure_timeout = configure_timeout
        self.jobs = jobs
        if not 1 <= jobs <= 64:
            raise ValueError("jobs must be between 1 and 64")

    def build(self, candidate: Candidate, training_input: Path | None = None) -> BuildResult:
        candidate = Candidate.from_dict(candidate.to_dict())
        build_dir = self.root / candidate.hash()
        result = BuildResult(candidate.hash(), candidate.to_dict(), str(build_dir))
        if build_dir.is_symlink():
            raise ValueError("build directory must not be a symbolic link")
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)
        try:
            if not (self.ledger / "CMakeLists.txt").is_file():
                raise ValueError("Ledger source is missing; initialize the pinned submodule")
            compiler = compiler_executable(candidate)
            version = run_command([compiler, "--version"], timeout=10,
                                  log=build_dir / "compiler-version.log")
            if not version.ok:
                raise RuntimeError(version.output)
            result.compiler_version = version.output.splitlines()[0]
            match = re.search(r"\b(\d+)\.", result.compiler_version)
            minimum = 13 if candidate.compiler == "gcc" else 17
            if not match or int(match.group(1)) < minimum:
                raise ValueError(f"{candidate.compiler} {minimum}+ is required: {result.compiler_version}")
            result.provenance.update(source_provenance(self.ledger))
            result.provenance.update({"compiler_executable": shutil.which(compiler),
                                      "configure_s": 0.0, "commands": [], "clean_build": True})
            profile = build_dir / "profile"
            phases = ("generate", "use") if candidate.pgo else ("normal",)
            if candidate.pgo:
                if training_input is None or not Path(training_input).is_file():
                    raise ValueError("PGO requires a separate existing training input")
                result.provenance["training_input_sha256"] = file_sha256(Path(training_input))
                result.provenance["training_input"] = str(Path(training_input).resolve())
                profile.mkdir()
            for phase in phases:
                flags = to_flags(candidate, phase=phase, profile_dir=profile)
                flags += (["-fsave-optimization-record"] if candidate.compiler == "clang"
                          else ["-fopt-info-missed"])
                result.flags = flags
                configure = ["cmake", "-S", self.ledger, "-B", build_dir,
                             "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_COMPILER=" + compiler,
                             "-DCMAKE_CXX_FLAGS=" + shlex.join(flags),
                             "-DCMAKE_CXX_FLAGS_RELEASE=-DNDEBUG", "-DLEDGER_NATIVE=OFF",
                             "-DBUILD_TESTING=ON", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"]
                if shutil.which("ninja"):
                    configure += ["-G", "Ninja"]
                configure_result = run_command(configure, timeout=self.configure_timeout,
                                               log=build_dir / f"configure-{phase}.log")
                result.provenance["configure_s"] += configure_result.elapsed_s
                result.provenance["commands"].append(configure_result.argv)
                if not configure_result.ok:
                    raise RuntimeError(configure_result.output)
                build_cmd = ["cmake", "--build", build_dir, "--parallel", str(self.jobs)]
                if phase == "generate":
                    build_cmd += ["--target", "ledger_bench"]
                built = run_command(build_cmd, timeout=self.build_timeout,
                                    log=build_dir / ("build.log" if phase != "generate" else "build-generate.log"))
                result.compile_s += built.elapsed_s
                result.provenance["commands"].append(built.argv)
                if not built.ok:
                    raise RuntimeError(built.output)
                if phase == "generate":
                    trained = run_command([build_dir / "ledger_bench", "--input",
                                           Path(training_input).resolve(), "--runs", "1",
                                           "--outdir", build_dir / "training"],
                                          timeout=self.build_timeout, log=build_dir / "training.log")
                    result.provenance["commands"].append(trained.argv)
                    if not trained.ok:
                        raise RuntimeError("PGO training failed: " + trained.output)
                    if candidate.compiler == "clang":
                        profiles = list(profile.rglob("*.profraw"))
                        if not profiles:
                            raise RuntimeError("Clang training produced no .profraw profiles")
                        merged = run_command(["llvm-profdata", "merge", "-o", profile / "merged.profdata",
                                              *profiles], timeout=self.build_timeout,
                                             log=build_dir / "profile-merge.log")
                        if not merged.ok:
                            raise RuntimeError("PGO merge failed: " + merged.output)
                    elif not list(profile.rglob("*.gcda")):
                        raise RuntimeError("GCC training produced no .gcda profiles")
            if candidate.pgo:
                from .profiles import verify_profiles
                verification = verify_profiles(candidate.compiler, build_dir, result.flags)
                result.provenance["profile_verification"] = verification
                if not verification["correctness"]:
                    raise RuntimeError("PGO profile-consumption gate failed: " + verification["reason"])
            required = [build_dir / name for name in ("ledger_bench", "ledger_differential")]
            if not all(path.is_file() for path in required):
                raise RuntimeError("Build succeeded without required benchmark/correctness binaries")
            sized = run_command(["size", "-A", build_dir / "ledger_bench"], timeout=10,
                                log=build_dir / "size.log")
            if sized.ok:
                from .feedback import parse_size
                result.text_bytes = parse_size(sized.output)
            else:
                result.provenance["text_size_unavailable"] = sized.output
            result.provenance["binary_sha256"] = file_sha256(build_dir / "ledger_bench")
            result.success = True
        except (OSError, ValueError, RuntimeError) as exc:
            result.error = str(exc)[-12288:]
        (build_dir / "build.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
        return result
