import json
from pathlib import Path

from pyrtc.loop import Loop
from pyrtc.science_camera import ScienceCamera
from pyrtc.slopes_process import SlopesProcess
from pyrtc.wavefront_corrector import WavefrontCorrector
from pyrtc.wavefront_sensor import WavefrontSensor
from pyrtc.utils import read_yaml_file


def test_pywfs_notebook_surrogate_smoke():
    repo_root = Path(__file__).resolve().parents[2]
    notebook_path = repo_root / "examples" / "pywfs" / "pywfs_example_OOPAO.ipynb"
    config_path = repo_root / "examples" / "pywfs" / "pywfs_OOPAO_config.yaml"
    param_path = repo_root / "examples" / "pywfs" / "pywfs_OOPAO_params.yaml"

    assert notebook_path.exists()
    assert config_path.exists()
    assert param_path.exists()

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    assert len(code_cells) > 0

    first_code_block = "\n".join("\n".join(cell.get("source", [])) for cell in code_cells[:4])
    assert 'CONFIG_PATH = Path("pywfs_OOPAO_config.yaml")' in first_code_block
    assert "read_yaml_file(str(CONFIG_PATH))" in first_code_block
    assert "pywfs_OOPAO_params.yaml" in first_code_block

    conf = read_yaml_file(str(config_path))
    param = read_yaml_file(str(param_path))
    for key in ("loop", "wfs", "wfc", "psf", "slopes"):
        assert key in conf
    for key in (
        "resolution",
        "diameter",
        "samplingTime",
        "ngs_band",
        "ngs_magnitude",
        "science_band",
        "science_magnitude",
        "r0",
        "L0",
        "nSubap",
        "modulation",
    ):
        assert key in param

    assert Loop is not None
    assert WavefrontSensor is not None
    assert WavefrontCorrector is not None
    assert SlopesProcess is not None
    assert ScienceCamera is not None
