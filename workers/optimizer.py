from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

MAX_OUTPUT_BYTES = 4096
RATIOS = ("1:1", "3:2", "2:3", "16:9", "9:16")


def _write_result(path: Path, result: dict[str, str]) -> None:
    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode()
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise ValueError("optimizer result exceeds 4096 bytes")
    path.write_bytes(encoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    request = json.loads(Path(args.input).read_text())
    prompt = request["prompt"]
    requested_ratio = request.get("aspect_ratio", "1:1")
    if requested_ratio not in RATIOS:
        requested_ratio = "1:1"

    model_path = os.environ["MODEL_PATH"]
    instruction = (
        "Rewrite the image-generation request as one precise English prompt. "
        "Preserve the user's subject, composition, text, mood, and constraints. "
        "Return JSON only with rewritten_prompt and wh_ratio. "
        f"wh_ratio must be one of {', '.join(RATIOS)} and should normally be "
        f"{requested_ratio}. Request: {prompt}"
    )
    schema = json.dumps(
        {
            "type": "object",
            "properties": {
                "rewritten_prompt": {"type": "string"},
                "wh_ratio": {"type": "string", "enum": list(RATIOS)},
            },
            "required": ["rewritten_prompt", "wh_ratio"],
            "additionalProperties": False,
        },
        separators=(",", ":"),
    )
    completed = subprocess.run(
        [
            "/app/llama-cli",
            "-m",
            model_path,
            "-p",
            instruction,
            "-n",
            "1024",
            "--temp",
            "0",
            "--seed",
            str(request.get("seed", 42)),
            "--json-schema",
            schema,
            "--no-display-prompt",
            "--no-warmup",
            "--simple-io",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    start = completed.stdout.find("{")
    end = completed.stdout.rfind("}")
    if start < 0 or end < start:
        raise ValueError("llama-cli did not return a JSON object")
    result = json.loads(completed.stdout[start : end + 1])
    if set(result) != {"rewritten_prompt", "wh_ratio"}:
        raise ValueError("optimizer returned unexpected fields")
    if not isinstance(result["rewritten_prompt"], str) or not result["rewritten_prompt"].strip():
        raise ValueError("optimizer returned an empty prompt")
    if result["wh_ratio"] not in RATIOS:
        raise ValueError("optimizer returned an unsupported ratio")
    _write_result(Path(args.output), result)


if __name__ == "__main__":
    main()
