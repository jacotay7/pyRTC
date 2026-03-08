import json
from pathlib import Path

from pyRTC.Loop import Loop
from pyRTC.ScienceCamera import ScienceCamera
from pyRTC.SlopesProcess import SlopesProcess
from pyRTC.WavefrontCorrector import WavefrontCorrector
from pyRTC.WavefrontSensor import WavefrontSensor
from pyRTC.utils import read_yaml_file


def test_pywfs_notebook_surrogate_smoke():
    repo_root = Path(__file__).resolve().parents[2]
    notebook_path = repo_root / "examples" / "scao" / "pywfs_example_OOPAO.ipynb"
    config_path = repo_root / "examples" / "scao" / "pywfs_OOPAO_config.yaml"

    assert notebook_path.exists()
    assert config_path.exists()

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    assert len(code_cells) > 0

    first_code_block = "\n".join("\n".join(cell.get("source", [])) for cell in code_cells[:4])
    assert "read_yaml_file(\"pywfs_OOPAO_config.yaml\")" in first_code_block

    conf = read_yaml_file(str(config_path))
    for key in ("loop", "wfs", "wfc", "psf", "slopes"):
        assert key in conf

    assert Loop is not None
    assert WavefrontSensor is not None
    assert WavefrontCorrector is not None
    assert SlopesProcess is not None
    assert ScienceCamera is not None
