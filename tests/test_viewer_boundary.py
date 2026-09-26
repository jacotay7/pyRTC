import os
import ast
from pathlib import Path
import subprocess
import sys


def _iter_viewer_python_files(repo_root: Path):
    viewer_dir = repo_root / "pyrtc" / "scripts"
    for path in sorted(viewer_dir.glob("*.py")):
        yield path


def test_viewer_cli_avoids_package_root_reexport_imports():
    repo_root = Path(__file__).resolve().parents[1]

    for viewer_file in _iter_viewer_python_files(repo_root):
        tree = ast.parse(viewer_file.read_text(encoding="utf-8"), filename=str(viewer_file))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module_name = node.module or ""
            assert module_name != "pyrtc", (
                f"{viewer_file.name} imports from the package root re-export surface. "
                "Viewer and CLI modules should import concrete submodules directly "
                "so they remain robust when pyrtc is resolved as a namespace package."
            )


def test_viewer_module_imports_when_repo_parent_is_on_sys_path():
    repo_root = Path(__file__).resolve().parents[1]
    # Run from the repo parent (so the repo root is not implicitly cwd[0]) while
    # putting the repo root on PYTHONPATH, exercising that view.py imports the
    # concrete submodules it needs rather than relying on cwd discovery.
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + existing if existing else "")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib; importlib.import_module('pyrtc.scripts.view'); print('ok')",
        ],
        cwd=repo_root.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert completed.stdout.strip() == "ok"
