import importlib.util
from pathlib import Path


def _load_diffuser():
    path = Path(__file__).parents[1] / "workers" / "diffuser.py"
    spec = importlib.util.spec_from_file_location("diffuser", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_diffuser_sizes_are_multiples_of_64():
    diffuser = _load_diffuser()
    assert diffuser.SIZES["1:1"] == (1024, 1024)
    assert all(width % 64 == height % 64 == 0 for width, height in diffuser.SIZES.values())


def test_diffuser_covers_supported_ratios():
    diffuser = _load_diffuser()
    assert set(diffuser.SIZES) == {"1:1", "3:2", "2:3", "16:9", "9:16"}
