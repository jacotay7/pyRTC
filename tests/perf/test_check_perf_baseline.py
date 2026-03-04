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
