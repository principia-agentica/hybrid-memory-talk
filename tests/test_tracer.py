import json
import time
from pathlib import Path

from tracing.tracer import Tracer


def _read_last_jsonl(path: Path) -> dict:
    text = path.read_text(encoding="utf-8").strip()
    assert text, "trace file should not be empty"
    line = text.splitlines()[-1]
    return json.loads(line)


def test_tracer_record_writes_jsonl_and_normalizes_ids(tmp_path):
    out_file = tmp_path / "traces.jsonl"
    tr = Tracer(path=str(out_file))

    inputs = "reset password for ana@example.com"
    retrieved = [
        {"id": "policy_pwd_01", "metadata": {"source": "policy.md", "section": "password"}},
        {"metadata": {"source": "policy.md", "section": "account"}},
        {"source": "episodic@t#user_turn"},
        "semantic#policy_pwd_02",
    ]
    output = "We sent a reset link."

    tr.record(inputs=inputs, retrieved=retrieved, output=output, span="qa")

    assert out_file.exists(), "trace file should be created lazily on first write"
    row = _read_last_jsonl(out_file)

    # Required fields
    required = [
        "ts",
        "span",
        "input_len",
        "ctx_len",
        "retrieved_ids",
        "output_len",
        "latency_ms",
    ]
    for key in required:
        assert key in row, f"missing field: {key}"

    # Shapes and values
    assert row["span"] == "qa"
    assert isinstance(row["retrieved_ids"], list)
    assert all(isinstance(x, str) for x in row["retrieved_ids"])  # normalized to strings
    assert row["ctx_len"] == len(row["retrieved_ids"]) == 4
    assert row["input_len"] == len(inputs)
    assert row["output_len"] == len(output)
    assert row["latency_ms"] >= 0


def test_tracer_span_latency_sanity(tmp_path):
    out_file = tmp_path / "trace.jsonl"
    tr = Tracer(path=str(out_file))

    sid = tr.start_span("retrieve", inputs="who am i?", ctx=[{"id": "x"}])
    time.sleep(0.01)  # ~10ms to ensure non-zero latency after rounding
    tr.end_span(sid, output=None, retrieved=[{"provenance": "episodic@y#event"}])

    row = _read_last_jsonl(out_file)
    assert row["span"] == "retrieve"
    assert row["latency_ms"] >= 5  # at least a few ms
    assert row["latency_ms"] < 2000  # sanity upper bound
    assert row["ctx_len"] == len(row["retrieved_ids"]) == 1
