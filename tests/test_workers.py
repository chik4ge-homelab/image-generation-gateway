import importlib.util
import json
from pathlib import Path

import pytest


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


def test_diffuser_cli_uses_model_and_generation_config_from_environment(monkeypatch):
    diffuser = _load_diffuser()
    arguments = [
        "--diffusion-model",
        "/models/custom-diffusion.gguf",
        "--vae",
        "/models/custom-vae.safetensors",
        "--llm",
        "/models/custom-prompt-model.gguf",
        "-p",
        "${PROMPT}",
        "--cfg-scale",
        "7",
        "--sampling-method",
        "heun",
        "--steps",
        "${STEPS}",
        "--seed",
        "${SEED}",
        "--backend",
        "diffusion=cuda0,te=cpu,vae=cuda0",
        "--params-backend",
        "diffusion=disk,te=cpu,vae=cuda0",
        "--max-vram",
        "cuda0=5",
        "--diffusion-fa",
        "-W",
        "${WIDTH}",
        "-H",
        "${HEIGHT}",
        "-o",
        "${OUTPUT}",
    ]
    monkeypatch.setenv("SD_CPP_ARGS", json.dumps(arguments))

    command = diffuser.build_sd_cli_command(
        {"prompt": "test prompt", "steps": 28, "seed": 123},
        1024,
        1024,
        Path("/tmp/image.png"),
    )

    assert command == [
        "/sd.cpp/bin/sd-cli",
        "--diffusion-model",
        "/models/custom-diffusion.gguf",
        "--vae",
        "/models/custom-vae.safetensors",
        "--llm",
        "/models/custom-prompt-model.gguf",
        "-p",
        "test prompt",
        "--cfg-scale",
        "7",
        "--sampling-method",
        "heun",
        "--steps",
        "28",
        "--seed",
        "123",
        "--backend",
        "diffusion=cuda0,te=cpu,vae=cuda0",
        "--params-backend",
        "diffusion=disk,te=cpu,vae=cuda0",
        "--max-vram",
        "cuda0=5",
        "--diffusion-fa",
        "-W",
        "1024",
        "-H",
        "1024",
        "-o",
        "/tmp/image.png",
    ]


def test_diffuser_cli_requires_model_config_in_environment(monkeypatch):
    diffuser = _load_diffuser()
    monkeypatch.delenv("SD_CPP_ARGS", raising=False)

    with pytest.raises(ValueError, match="SD_CPP_ARGS"):
        diffuser.build_sd_cli_command({"prompt": "test"}, 1024, 1024, Path("/tmp/image.png"))
