"""First Loop iteration after construction runs near steady-state speed.

Each run happens in a fresh interpreter, because earlier tests in this
process have already compiled the kernels and would hide a stall.
"""

import json

from benchmarks import first_iteration_bench as bench


def test_first_loop_iteration_is_not_a_compile_stall():
    row = bench.run_case_in_subprocess(
        "loop.leaky_integrator", iterations=60, signal_size=400, num_modes=100, idle=0.05
    )

    # Without the warm-up the first call compiles the kernel or loads it from
    # the numba cache: about 0.15 s with a warm cache and 0.7 s cold, over
    # 1000x the steady state. Single latencies on shared CI runners are
    # noisy (scheduler, thread-pool wake-up), so the bound is generous: it
    # passes when either the ratio or the absolute time is small.
    assert row["first_s"] < 0.05 or row["first_over_median"] < 50, row


def test_main_writes_json(tmp_path):
    output = tmp_path / "first_iteration.json"

    code = bench.main(
        [
            "--cases",
            "wfc.send_to_hardware",
            "--iterations",
            "3",
            "--num-modes",
            "16",
            "--idle",
            "0",
            "--output",
            str(output),
        ]
    )

    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["meta"]["benchmark_type"] == "first_iteration"
    rows = payload["results"]["wfc.send_to_hardware"]
    assert set(rows) == {"cold_cache", "warm_cache"}
    assert rows["warm_cache"]["first_s"] > 0
