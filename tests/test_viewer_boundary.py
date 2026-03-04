import ast
from pathlib import Path


def _iter_viewer_python_files(repo_root: Path):
    viewer_dir = repo_root / "pyRTC" / "scripts"
    for path in sorted(viewer_dir.glob("*.py")):
        yield path


def test_viewer_cli_uses_public_pyrtc_imports_only():
    repo_root = Path(__file__).resolve().parents[1]
    disallowed_prefixes = ("pyRTC.Pipeline", "pyRTC.utils")

    for viewer_file in _iter_viewer_python_files(repo_root):
        tree = ast.parse(viewer_file.read_text(encoding="utf-8"), filename=str(viewer_file))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module_name = node.module or ""
            assert not module_name.startswith(disallowed_prefixes), (
                f"{viewer_file.name} imports internal module '{module_name}'. "
                "Use top-level pyRTC imports instead."
            )
