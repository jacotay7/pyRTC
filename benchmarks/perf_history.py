"""Performance history: trends and regressions across CI runs or saved reports.

CI uploads a ``perf-smoke-report-py<version>`` artifact on every run (pull
requests into ``dev``/``main`` and pushes to ``main``), and a
``pipeline-latency`` artifact from one Python version. This tool collects
those reports, from GitHub (``--repo``, needs a token in ``GH_TOKEN`` or
``GITHUB_TOKEN``) or from a directory of saved JSON reports (``--from-dir``,
e.g. a lab host running ``perf_smoke.py`` nightly). It then prints each
metric's latest value against the median of the earlier runs and flags
regressions::

    python -m benchmarks.perf_history --repo jacotay7/pyRTC --runs 20
    python -m benchmarks.perf_history --from-dir lab_reports/ --plot trends.png
    python -m benchmarks.perf_history --repo jacotay7/pyRTC --current perf_smoke_report.json

Exits 1 when any metric's latest value exceeds ``--max-ratio`` times its
history median (0 otherwise), so it can gate a scheduled job.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import statistics
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable

API = "https://api.github.com"
# Preferred timing field of a measurement, and its scale to seconds.
TIME_KEYS = (("median_s", 1.0), ("mean_s", 1.0), ("p50_us", 1e-6), ("mean_us", 1e-6))


def _item_label(item: Any, index: int) -> str:
    """Name a list entry by its identifying fields (pipeline latency summaries)."""

    if not isinstance(item, dict):
        return str(index)
    parts = []
    if "mode" in item:
        parts.append(str(item["mode"]))
    if "notify" in item:
        parts.append("notify" if item["notify"] else "poll")
    if "source" in item and "target" in item:
        parts.append(f"{item['source']}->{item['target']}")
    return ",".join(parts) or str(index)


def flatten_report(report: Any, prefix: str = "") -> dict[str, float]:
    """Return ``{metric path: seconds}`` for every timed entry in a report.

    A timed entry is a mapping with ``median_s``, ``mean_s``, ``p50_us`` or
    ``mean_us`` (the first present wins). Paths join the nested keys with
    ``/``; list entries are named by their ``mode``/``notify``/
    ``source->target`` fields, or their index.
    """

    metrics: dict[str, float] = {}
    if isinstance(report, dict):
        for key, scale in TIME_KEYS:
            value = report.get(key)
            if isinstance(value, (int, float)) and value > 0:
                metrics[prefix.rstrip("/")] = float(value) * scale
                return metrics
        for key, value in report.items():
            metrics.update(flatten_report(value, f"{prefix}{key}/"))
    elif isinstance(report, list):
        for index, item in enumerate(report):
            metrics.update(flatten_report(item, f"{prefix}{_item_label(item, index)}/"))
    return metrics


def compare(history: list[dict[str, float]], *, window: int = 10) -> list[dict[str, Any]]:
    """Compare the newest run's metrics with the median of up to ``window`` earlier runs.

    ``history`` is ordered oldest first. Returns one row per metric present
    in the newest run and at least one earlier run, sorted worst first.
    """

    if len(history) < 2:
        return []
    latest, earlier = history[-1], history[:-1][-window:]
    rows = []
    for name, value in latest.items():
        previous = [run[name] for run in earlier if name in run]
        if not previous:
            continue
        baseline = statistics.median(previous)
        rows.append(
            {
                "metric": name,
                "latest_s": value,
                "median_s": baseline,
                "ratio": value / baseline if baseline > 0 else float("inf"),
                "runs": len(previous),
            }
        )
    rows.sort(key=lambda row: row["ratio"], reverse=True)
    return rows


def format_table(rows: Iterable[dict[str, Any]], *, max_ratio: float, limit: int = 25) -> str:
    lines = [
        "| metric | latest | history median | ratio | runs |",
        "|---|---|---|---|---|",
    ]
    for row in list(rows)[:limit]:
        flag = " **REGRESSION**" if row["ratio"] > max_ratio else ""
        lines.append(
            f"| {row['metric']} | {row['latest_s'] * 1e6:.1f} us | {row['median_s'] * 1e6:.1f} us "
            f"| {row['ratio']:.2f}{flag} | {row['runs']} |"
        )
    return "\n".join(lines)


# -- sources --------------------------------------------------------------------


def load_directory(directory: str | Path) -> list[tuple[str, dict[str, float]]]:
    """Load ``*.json`` reports from ``directory``, oldest first.

    Reports are ordered by their ``timestamp_unix`` field when present, else
    by file name.
    """

    runs = []
    for path in sorted(Path(directory).glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        runs.append((float(report.get("timestamp_unix", 0.0)), path.name, report))
    runs.sort(key=lambda item: (item[0], item[1]))
    return [(name, flatten_report(report)) for _, name, report in runs]


class _DropAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Artifact downloads redirect to blob storage, which rejects GitHub tokens."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if (
            new is not None
            and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc
        ):
            new.remove_header("Authorization")
        return new


_OPENER = urllib.request.build_opener(_DropAuthOnRedirect)


def _request(url: str, token: str | None, *, raw: bool = False):
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with _OPENER.open(request, timeout=60) as response:
        body = response.read()
    return body if raw else json.loads(body)


def load_github(
    repo: str,
    *,
    branch: str | None = None,
    workflow: str = "python-install.yml",
    artifact_prefix: str = "perf-smoke-report-py3.12",
    runs: int = 20,
    token: str | None = None,
    request=_request,
) -> list[tuple[str, dict[str, float]]]:
    """Download the perf artifacts of the latest successful runs, oldest first.

    CI runs on pull requests into ``dev``/``main`` and on pushes to ``main``,
    so by default every successful run counts; ``branch`` limits it to runs
    whose head is that branch.
    """

    query = f"status=success&per_page={int(runs)}" + (f"&branch={branch}" if branch else "")
    listing = request(f"{API}/repos/{repo}/actions/workflows/{workflow}/runs?{query}", token)
    collected = []
    for run in listing.get("workflow_runs", []):
        artifacts = request(f"{API}/repos/{repo}/actions/runs/{run['id']}/artifacts", token)
        metrics: dict[str, float] = {}
        for artifact in artifacts.get("artifacts", []):
            if artifact.get("expired") or not artifact["name"].startswith(artifact_prefix):
                continue
            archive = zipfile.ZipFile(
                io.BytesIO(request(artifact["archive_download_url"], token, raw=True))
            )
            for member in archive.namelist():
                if member.endswith(".json"):
                    metrics.update(flatten_report(json.loads(archive.read(member))))
        if metrics:
            label = f"{run.get('created_at', '')} {run.get('head_sha', '')[:7]}"
            collected.append((run.get("created_at", ""), label, metrics))
    collected.sort(key=lambda item: item[0])
    return [(label, metrics) for _, label, metrics in collected]


def plot(
    runs: list[tuple[str, dict[str, float]]], rows: list[dict[str, Any]], path: str, count: int = 8
):
    """Plot the ``count`` worst metrics over the runs (needs matplotlib)."""

    from pyrtc.utils import pyplot

    plt = pyplot()
    figure, axis = plt.subplots(figsize=(10, 5))
    for row in rows[:count]:
        series = [metrics.get(row["metric"]) for _, metrics in runs]
        axis.plot(
            range(len(runs)),
            [s * 1e6 if s else None for s in series],
            marker="o",
            label=row["metric"],
        )
    axis.set_xlabel("run (oldest to newest)")
    axis.set_ylabel("time (us)")
    axis.set_yscale("log")
    axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repo", help="GitHub owner/name to read CI artifacts from")
    source.add_argument("--from-dir", help="Directory of saved perf JSON reports")
    parser.add_argument("--branch", help="Only runs whose head is this branch (default: all)")
    parser.add_argument("--workflow", default="python-install.yml")
    parser.add_argument(
        "--artifact",
        default="perf-smoke-report-py3.12",
        help="Artifact name prefix (pipeline-latency for the end-to-end benchmark)",
    )
    parser.add_argument("--runs", type=int, default=20, help="How many recent runs to read")
    parser.add_argument(
        "--current",
        nargs="+",
        help="Report(s) from this run, added as the newest run (e.g. in CI before upload)",
    )
    parser.add_argument(
        "--window", type=int, default=10, help="Earlier runs in the baseline median"
    )
    parser.add_argument("--max-ratio", type=float, default=1.5, help="Regression threshold")
    parser.add_argument("--plot", help="Write a trend plot of the worst metrics to this file")
    parser.add_argument("--json", help="Write the comparison rows to this JSON file")
    args = parser.parse_args(argv)

    if args.repo:
        token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        runs = load_github(
            args.repo,
            branch=args.branch,
            workflow=args.workflow,
            artifact_prefix=args.artifact,
            runs=args.runs,
            token=token,
        )
    else:
        runs = load_directory(args.from_dir)
    if args.current:
        current: dict[str, float] = {}
        for path in args.current:
            current.update(flatten_report(json.loads(Path(path).read_text(encoding="utf-8"))))
        runs.append(("current", current))
    if len(runs) < 2:
        print(f"Need at least two runs with reports; found {len(runs)}.")
        return 0
    rows = compare([metrics for _, metrics in runs], window=args.window)
    print(f"{len(runs)} runs, latest: {runs[-1][0]}")
    print(format_table(rows, max_ratio=args.max_ratio))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if args.plot:
        plot(runs, rows, args.plot)
    regressions = [row for row in rows if row["ratio"] > args.max_ratio]
    if regressions:
        print(f"{len(regressions)} metric(s) regressed beyond {args.max_ratio}x.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
