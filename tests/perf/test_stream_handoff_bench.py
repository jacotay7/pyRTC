"""Smoke test for the stream handoff (notify on/off) benchmark."""

import pyshmem
import pytest

from benchmarks import stream_handoff_bench as bench


@pytest.mark.parametrize("mode", ["thread", "process"])
@pytest.mark.parametrize("notify", [False, True])
def test_run_handoff_smoke(mode, notify):
    before = set(pyshmem.list_streams())
    row = bench.run_handoff(mode, notify, samples=5, period=1e-3)

    assert row["mode"] == mode
    assert row["notify"] is notify
    assert row["handoff"]["count"] == 5
    assert row["handoff"]["p50_us"] > 0
    assert row["writer_write_call"]["count"] >= 5
    assert 0.0 <= row["consumer_cpu_fraction"]
    leaked = {name for name in set(pyshmem.list_streams()) - before if "_handoff" in name}
    assert not leaked
