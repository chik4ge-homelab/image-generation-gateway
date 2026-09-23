import importlib.util
import subprocess
import tempfile
from pathlib import Path


def _load(name: str):
    path = Path(__file__).parents[1] / "workers" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_diffuser_sizes_are_multiples_of_64():
    diffuser = _load("diffuser")
    assert diffuser.SIZES["1:1"] == (1024, 1024)
    assert all(width % 64 == height % 64 == 0 for width, height in diffuser.SIZES.values())


def test_optimizer_and_diffuser_support_the_same_ratios():
    optimizer = _load("optimizer")
    diffuser = _load("diffuser")
    assert set(optimizer.RATIOS) == set(diffuser.SIZES)


def test_optimizer_has_bounded_gpu_inference():
    optimizer = _load("optimizer")
    assert optimizer.MAX_TOKENS == "512"
    assert optimizer.CONTEXT_SIZE == "1024"
    assert optimizer.GPU_LAYERS == "99"
    assert optimizer.THREADS == "2"
    assert optimizer.BATCH_SIZE == "128"
    assert optimizer.UBATCH_SIZE == "128"
    assert optimizer.TIMEOUT_SECONDS == 300


def test_optimizer_relies_on_model_json_instead_of_llama_grammar():
    source = (Path(__file__).parents[1] / "workers" / "optimizer.py").read_text()
    assert "--json-schema" not in source


def test_optimizer_does_not_buffer_llama_stderr(monkeypatch):
    optimizer = _load("optimizer")
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="{}")

    monkeypatch.setattr(optimizer.subprocess, "run", fake_run)
    optimizer._run_llama(["llama-cli"])
    assert captured["stdout"] is subprocess.PIPE
    assert "stderr" not in captured
    assert "capture_output" not in captured


def test_optimizer_uses_llm_gateway_memory_tuning(monkeypatch):
    optimizer = _load("optimizer")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(
            command, 0, stdout='{"rewritten_prompt":"x","wh_ratio":"1:1"}'
        )

    monkeypatch.setattr(optimizer.subprocess, "run", fake_run)
    monkeypatch.setenv("MODEL_PATH", "/models/model.gguf")
    with tempfile.TemporaryDirectory() as directory:
        input_path = Path(directory) / "input.json"
        output_path = Path(directory) / "output.json"
        input_path.write_text('{"prompt":"cat"}')
        monkeypatch.setattr(
            "sys.argv",
            ["optimizer", "--input", str(input_path), "--output", str(output_path)],
        )
        optimizer.main()

    command = captured["command"]
    for pair in (
        ("--no-mmap", "-ngl"),
        ("--flash-attn", "on"),
        ("-ctk", "q8_0"),
        ("-ctv", "q8_0"),
        ("-b", "128"),
        ("-ub", "128"),
    ):
        offset = command.index(pair[0])
        if pair[0] == "--no-mmap":
            assert command[offset + 1] == pair[1]
        else:
            assert list(pair) == command[offset : offset + 2]
