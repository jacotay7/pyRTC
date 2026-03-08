import json

from benchmarks import perf_smoke


def test_run_perf_smoke_schema(monkeypatch):
    called = {"value": False}

    def _fake_core_compute(**kwargs):
        called["value"] = True
        return {"kernels": {"dummy": {"mean_s": 0.0}}, "gpu_kernels": {"status": {"available": False}}}

    monkeypatch.setattr(perf_smoke, "run_core_compute_benchmarks", _fake_core_compute)

    report = perf_smoke.run_perf_smoke(num_iters=5, num_samples=100)

    assert "platform" in report
    assert "python" in report
    assert "measure_execution_time" in report
    assert "latency_math" in report
    assert "core_compute" in report
    assert called["value"] is True

    met = report["measure_execution_time"]
    assert met["median"] >= 0
    assert met["ci1"] <= met["ci99"]

    lat = report["latency_math"]
    assert lat["frame_shift"] >= 0
    assert lat["mean_latency_s"] >= 0


def test_perf_smoke_main_writes_json(tmp_path):
    output = tmp_path / "perf.json"
    code = perf_smoke.main(
        [
            "--output",
            str(output),
            "--num-iters",
            "3",
            "--num-samples",
            "50",
            "--skip-core",
        ]
    )

    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "measure_execution_time" in payload
    assert "latency_math" in payload


def test_run_perf_smoke_skip_core():
    report = perf_smoke.run_perf_smoke(num_iters=3, num_samples=50, include_core=False)
    assert "core_compute" not in report
