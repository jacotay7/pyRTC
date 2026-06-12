from benchmarks import check_perf_baseline


def _sample_report(include_gpu=False):
    report = {
        "core_compute": {
            "profiles": {
                "10x10": {
                    "k": {
                        "mean_s": 0.1,
                        "median_s": 0.1,
                        "p95_s": 0.2,
                        "p99_s": 0.3,
                        "p99_hz": 3.3,
                    }
                }
            }
        },
    }
    if include_gpu:
        report["core_compute"]["gpu_kernels"] = {
            "status": {"available": True},
            "g": {
                "mean_s": 0.1,
                "median_s": 0.1,
                "p95_s": 0.2,
                "p99_s": 0.3,
                "p99_hz": 3.3,
            },
        }
    return report


def test_compare_against_baseline_success():
    current = _sample_report(include_gpu=True)
    baseline = _sample_report(include_gpu=True)["core_compute"]

    missing, comparison = check_perf_baseline.compare_against_baseline(current, baseline)

    assert missing == []
    assert "core_compute.profiles.10x10.k.mean_s" in comparison


def test_compare_against_baseline_detects_missing_metric():
    current = _sample_report(include_gpu=False)
    baseline = _sample_report(include_gpu=False)
    del baseline["core_compute"]["profiles"]["10x10"]["k"]["p99_hz"]

    missing, _ = check_perf_baseline.compare_against_baseline(current, baseline)

    assert "core_compute.profiles.10x10.k.p99_hz" in missing


def _sample_closed_loop_report(include_gpu=True):
    report = {
        "meta": {
            "benchmark_type": "synthetic_closed_loop",
        },
        "results": {
            "pywfs": {
                "10x10": {
                    "cpu": {
                        "mean_s": 0.1,
                        "median_s": 0.1,
                        "p95_s": 0.2,
                        "p99_s": 0.3,
                        "p99_hz": 3.3,
                    },
                }
            }
        },
    }
    if include_gpu:
        report["results"]["pywfs"]["10x10"]["gpu"] = {
            "mean_s": 0.05,
            "median_s": 0.05,
            "p95_s": 0.1,
            "p99_s": 0.15,
            "p99_hz": 6.6,
        }
    return report


def test_compare_against_baseline_supports_closed_loop_reports():
    current = _sample_closed_loop_report(include_gpu=True)
    baseline = _sample_closed_loop_report(include_gpu=True)

    missing, comparison = check_perf_baseline.compare_against_baseline(current, baseline)

    assert missing == []
    assert "results.pywfs.10x10.cpu.mean_s" in comparison
    assert "results.pywfs.10x10.gpu.p99_hz" in comparison


def test_compare_against_baseline_detects_missing_closed_loop_metric():
    current = _sample_closed_loop_report(include_gpu=False)
    baseline = _sample_closed_loop_report(include_gpu=False)
    del baseline["results"]["pywfs"]["10x10"]["cpu"]["p99_hz"]

    missing, _ = check_perf_baseline.compare_against_baseline(current, baseline)

    assert "results.pywfs.10x10.cpu.p99_hz" in missing


def test_find_ratio_regressions_flags_slow_latency_and_low_throughput():
    from benchmarks.check_perf_baseline import find_ratio_regressions

    comparison = {
        "a.mean_s": {"current": 2.0, "baseline": 1.0, "ratio": 2.0},
        "a.p99_hz": {"current": 100.0, "baseline": 1000.0, "ratio": 0.1},
        "b.mean_s": {"current": 1.1, "baseline": 1.0, "ratio": 1.1},
        "b.p99_hz": {"current": 900.0, "baseline": 1000.0, "ratio": 0.9},
        "c.count": {"current": 5, "baseline": 5, "ratio": 1.0},
    }

    regressions = find_ratio_regressions(comparison, max_ratio=1.5)

    assert set(regressions) == {"a.mean_s", "a.p99_hz"}


def test_find_ratio_regressions_rejects_invalid_threshold():
    import pytest

    from benchmarks.check_perf_baseline import find_ratio_regressions

    with pytest.raises(ValueError):
        find_ratio_regressions({}, max_ratio=1.0)


def test_main_fails_on_ratio_regression(tmp_path):
    import json

    from benchmarks.check_perf_baseline import main

    report = {
        "core_compute": {
            "profiles": {
                "cpu": {
                    "kernel": {
                        "mean_s": 2.0,
                        "median_s": 2.0,
                        "p95_s": 2.0,
                        "p99_s": 2.0,
                        "p99_hz": 0.5,
                    }
                }
            }
        }
    }
    baseline = json.loads(json.dumps(report))
    for metric in baseline["core_compute"]["profiles"]["cpu"]["kernel"]:
        baseline["core_compute"]["profiles"]["cpu"]["kernel"][metric] = 1.0

    current_path = tmp_path / "current.json"
    baseline_path = tmp_path / "baseline.json"
    current_path.write_text(json.dumps(report), encoding="utf-8")
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    import pytest

    with pytest.raises(SystemExit, match="Performance regression"):
        main(
            ["--current", str(current_path), "--baseline", str(baseline_path), "--max-ratio", "1.5"]
        )

    # presence-only mode still passes
    assert main(["--current", str(current_path), "--baseline", str(baseline_path)]) == 0
