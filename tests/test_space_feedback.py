import random
from pathlib import Path

import pytest

from compilerpilot.space import Candidate, compiler_executable, list_knobs, sample_candidate, to_flags
from compilerpilot.feedback import parse_gcc_feedback, parse_optimization_records, parse_perf, parse_size, summarize_feedback


@pytest.mark.parametrize("payload", [
    {"flags": "-Ofast"}, {"compiler": "g++;touch /tmp/unsafe"}, {"opt": "Ofast"},
    {"lto": 1}, {"inline": True}, {"alignment": False}, {"isa": "native;echo pwn"},
    {"alignment": "32"}, {"inline": "500"}, {"vectorize": None},
])
def test_rejects_arbitrary_flags_and_wrong_types(payload):
    with pytest.raises(ValueError):
        Candidate.from_dict(payload)


def test_hash_roundtrip_and_strict_baseline():
    baseline = Candidate()
    assert baseline == Candidate.from_dict(baseline.to_dict())
    assert baseline.hash() == Candidate.from_dict(dict(reversed(list(baseline.to_dict().items())))).hash()
    assert to_flags(baseline) == ["-O3", "-DNDEBUG", "-march=native"]
    assert compiler_executable(baseline) == "g++"
    knobs = list_knobs()
    knobs["opt"].append("Ofast")
    assert "Ofast" not in list_knobs()["opt"]


def test_compiler_specific_allowlisted_flags(tmp_path):
    gcc = Candidate(inline=100, vectorize=False, lto=True, pgo=True, alignment=32,
                    frame_pointer=True, unroll=True, semantic_interposition=False)
    flags = to_flags(gcc, "generate", tmp_path)
    assert "-flto" in flags and "-finline-limit=100" in flags
    assert "-fno-tree-vectorize" in flags and f"-fprofile-generate={tmp_path}" in flags
    use = to_flags(gcc, "use", tmp_path)
    assert f"-fprofile-use={tmp_path}" in use
    clang = Candidate(compiler="clang", pgo=True, lto=True, inline=500, vectorize=False)
    assert "-flto=thin" in to_flags(clang, "generate", tmp_path)
    assert "-fuse-ld=lld" in to_flags(clang, "generate", tmp_path)
    assert f"-fprofile-instr-use={tmp_path}/merged.profdata" in to_flags(clang, "use", tmp_path)
    assert "-inline-threshold=500" in to_flags(clang, "use", tmp_path)
    with pytest.raises(ValueError):
        to_flags(clang)
    with pytest.raises(ValueError):
        to_flags(Candidate(), "generate", tmp_path)


def test_sampling_is_reproducible_and_constrained():
    first = random.Random(12)
    second = random.Random(12)
    assert [sample_candidate(first).to_dict() for _ in range(20)] == [sample_candidate(second).to_dict() for _ in range(20)]
    assert all(sample_candidate(first, ("gcc",)).compiler == "gcc" for _ in range(20))
    with pytest.raises(ValueError):
        sample_candidate(first, ("shell",))


def test_tagged_yaml_filters_standard_library_and_limits_lists():
    records = [f'''--- !Missed
Pass: inline
Name: TooCostly
DebugLoc: {{ File: /work/ledger/src/book.cpp, Line: {i} }}
Function: Book_match
Hotness: {i}
Args:
  - String: 'too costly'
...\n''' for i in range(15)]
    records.append('''--- !Missed
Pass: loop-vectorize
DebugLoc: { File: /usr/include/c++/vector, Line: 1 }
Function: std_vector
Args: []
...\n''')
    records.append('''--- !Missed
Pass: loop-vectorize
DebugLoc: { File: include/ledger/book.hpp, Line: 8 }
Function: Book_cancel
Args:
  - String: 'unsafe dependent operation'
...\n''')
    parsed = parse_optimization_records("".join(records))
    assert parsed["missed_inline_count"] == 15
    assert len(parsed["missed_inline"]) == 10
    assert parsed["missed_inline"][0]["hotness"] == 14
    assert parsed["missed_vectorize"] == 1
    assert parsed["parse_errors"] == []


def test_yaml_malformed_feedback_is_explicit():
    assert parse_optimization_records("--- !Missed\nPass: [broken")["parse_errors"]


def test_gcc_filters_external_sources():
    parsed = parse_gcc_feedback('''/home/ledger/src/book.cpp:3:2: missed: not inlinable: Book::cancel
/usr/include/c++/vector:2:1: missed: couldn't vectorize loop
include/ledger/id_index.hpp:5:1: missed: couldn't vectorize loop
''')
    assert parsed["missed_inline_count"] == 1
    assert parsed["missed_vectorize"] == 1


def test_perf_human_csv_and_unavailable():
    human = parse_perf("1,000 cycles\n3,000 instructions # 3.00 insn per cycle\n10 branch-misses\n100 branches\n<not supported> L1-icache-load-misses")
    assert human["ipc"] == 3.0 and human["branch_miss_pct"] == 10.0
    assert human["icache_misses"] is None
    assert human["unavailable_events"] == ["L1-icache-load-misses"]
    csv = parse_perf("1000;;cycles;1;100.00\n2500;;instructions;1;100.00\n")
    assert csv["ipc"] == 2.5
    absent = parse_perf("Error: No permission to enable cycles event")
    assert not absent["available"] and absent["cycles"] is None
    assert absent["reason"]


def test_size_uses_text_only():
    assert parse_size(".text 1024 4096\n.text.hot 2048 8192\n.data 999 0\n") == 3072
    with pytest.raises(ValueError):
        parse_size("data bss dec\n100 200 300")


def test_summarizer_marks_missing_observability(tmp_path):
    summary = summarize_feedback(tmp_path)
    assert not summary["remarks_available"]
    assert not summary["counters"]["available"]
    assert summary["ipc"] is None


def test_summarizer_reads_nested_perf_counter_log(tmp_path):
    (tmp_path / "counters").mkdir()
    (tmp_path / "counters" / "perf.log").write_text("1000;;cycles;1;100.00\n3000;;instructions;1;100.00\n")
    summary = summarize_feedback(tmp_path)
    assert summary["counters"]["available"]
    assert summary["ipc"] == 3.0
