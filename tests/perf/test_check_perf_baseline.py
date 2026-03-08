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
