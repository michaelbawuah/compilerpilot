from pathlib import Path
import subprocess

import pytest

from compilerpilot.profiles import CORE_SOURCES, REQUIRED_LLVM_FUNCTIONS, profile_diagnostics, verify_profiles


def _clang_fixture(tmp_path, monkeypatch, counts=12):
    profile = tmp_path / "profile" / "merged.profdata"
    profile.parent.mkdir()
    profile.write_bytes(b"fixture-profile")
    (tmp_path / "build.log").write_text("[1/2] Building CXX object ledger.dir/src/book.cpp.o\n")
    output = "Counters:\n" + "".join(
        f"  {symbol}:\n    Hash: 0x1\n    Counters: 1\n    Function count: {counts}\n    Block counts: []\n"
        for symbol in REQUIRED_LLVM_FUNCTIONS.values()
    )
    monkeypatch.setattr("compilerpilot.profiles.subprocess.run", lambda *a, **k:
                        subprocess.CompletedProcess(a[0], 0, stdout=output, stderr=""))
    return [f"-fprofile-instr-use={profile}"]


def _gcc_fixture(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    for source in CORE_SOURCES:
        (profile / ("#build#CMakeFiles#ledger.dir#src#" + source + ".gcda")).write_bytes(
            b"adcg" + bytes(28))
    (tmp_path / "build.log").write_text("")
    return [f"-fprofile-use={profile}"]


def test_llvm_positive_hot_path_counts_and_hashes(tmp_path, monkeypatch):
    result = verify_profiles("clang", tmp_path, _clang_fixture(tmp_path, monkeypatch))
    assert result["correctness"]
    assert result["counts"] == {name: 12 for name in REQUIRED_LLVM_FUNCTIONS}
    assert len(next(iter(result["profile_hashes"].values()))) == 64


def test_llvm_unexecuted_core_is_rejected(tmp_path, monkeypatch):
    result = verify_profiles("clang", tmp_path, _clang_fixture(tmp_path, monkeypatch, counts=0))
    assert not result["correctness"]
    assert "positive execution counts" in result["reason"]


def test_llvm_tool_error_is_rejected(tmp_path, monkeypatch):
    flags = _clang_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr("compilerpilot.profiles.subprocess.run", lambda *a, **k:
                        subprocess.CompletedProcess(a[0], 1, stdout="", stderr="bad profile"))
    assert not verify_profiles("clang", tmp_path, flags)["correctness"]


@pytest.mark.parametrize("compiler", ["gcc", "clang"])
def test_missing_log_is_rejected(tmp_path, compiler):
    assert not verify_profiles(compiler, tmp_path, [])["correctness"]


def test_gcc_requires_all_trained_core_profiles(tmp_path):
    flags = _gcc_fixture(tmp_path)
    assert verify_profiles("gcc", tmp_path, flags)["correctness"]
    next((tmp_path / "profile").glob("*bench_replay.cpp.gcda")).unlink()
    result = verify_profiles("gcc", tmp_path, flags)
    assert not result["correctness"]
    assert "bench_replay.cpp" in result["reason"]


def test_gcc_rejects_corrupt_profile(tmp_path):
    flags = _gcc_fixture(tmp_path)
    next((tmp_path / "profile").glob("*book.cpp.gcda")).write_bytes(b"not-gcda")
    assert not verify_profiles("gcc", tmp_path, flags)["correctness"]


@pytest.mark.parametrize("source", CORE_SOURCES)
def test_core_warning_rejected_even_with_profile_present(tmp_path, source):
    flags = _gcc_fixture(tmp_path)
    (tmp_path / "build.log").write_text(
        f"/ledger/src/{source}:4:1: warning: profile count data file not found [-Wmissing-profile]\n")
    assert not verify_profiles("gcc", tmp_path, flags)["correctness"]


def test_untrained_auxiliary_warning_is_recorded(tmp_path):
    flags = _gcc_fixture(tmp_path)
    (tmp_path / "build.log").write_text(
        "/ledger/tests/engine_tests.cpp:4:1: warning: profile count data file not found [-Wmissing-profile]\n")
    result = verify_profiles("gcc", tmp_path, flags)
    assert result["correctness"]
    assert result["diagnostics"][0]["source"] == "engine_tests.cpp"


def test_clang_context_and_unknown_warning_fail_closed():
    lines = ('[1/2] Building CXX object ledger.dir/src/book.cpp.o\n'
             'warning: profile data may be out of date: mismatched data ignored [-Wprofile-instr-out-of-date]\n')
    assert profile_diagnostics(lines)[0]["core"]
    assert profile_diagnostics('warning: no profile data available [-Wprofile-instr-unprofiled]')[0]["unassigned"]


def test_profile_path_cannot_escape_build(tmp_path):
    (tmp_path / "build.log").write_text("")
    result = verify_profiles("gcc", tmp_path, [f"-fprofile-use={tmp_path.parent}"])
    assert not result["correctness"]
    assert "inside this build" in result["reason"]


def test_generation_flags_cannot_qualify_final_build(tmp_path):
    flags = _gcc_fixture(tmp_path) + ["-fprofile-generate=/tmp/ignored"]
    assert not verify_profiles("gcc", tmp_path, flags)["correctness"]
