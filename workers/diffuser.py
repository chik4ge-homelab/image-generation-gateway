from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from string import Template

import boto3

MAX_OUTPUT_BYTES = 4096
SIZES = {
    "1:1": (1024, 1024),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
}


def _endpoint() -> str:
    endpoint = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
    if endpoint:
        return endpoint
    configured = os.environ.get("ARTIFACT_ENDPOINT")
    if configured:
        return configured
    host = os.environ["BUCKET_HOST"]
    port = os.environ.get("BUCKET_PORT")
    endpoint = host if "://" in host else f"http://{host}"
    return f"{endpoint}:{port}" if port else endpoint


def _required_config(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"missing required configuration: {name}")
    return value


def build_sd_cli_command(
    request: dict, width: int, height: int, output_path: Path
) -> list[str]:
    try:
        arguments = json.loads(_required_config("SD_CPP_ARGS"))
    except json.JSONDecodeError as exc:
        raise ValueError("SD_CPP_ARGS must be a JSON array of argument templates") from exc
    if not isinstance(arguments, list) or not all(isinstance(arg, str) for arg in arguments):
        raise ValueError("SD_CPP_ARGS must be a JSON array of argument templates")

    values = {
        "PROMPT": request["prompt"],
        "STEPS": str(request.get("steps", 40)),
        "SEED": str(request.get("seed", 42)),
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "OUTPUT": str(output_path),
    }
    try:
        expanded = [Template(arg).substitute(values) for arg in arguments]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid placeholder in SD_CPP_ARGS: {exc}") from exc
    return ["/sd.cpp/bin/sd-cli", *expanded]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    request = json.loads(Path(args.input).read_text())
    width, height = SIZES[request["wh_ratio"]]
    with tempfile.TemporaryDirectory() as output_dir:
        image_path = Path(output_dir) / "image.png"
        command = build_sd_cli_command(request, width, height, image_path)
        subprocess.run(command, check=True)
        payload = image_path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        client = boto3.client(
            "s3",
            endpoint_url=_endpoint(),
            region_name=os.environ.get("BUCKET_REGION") or "us-east-1",
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
        client.put_object(
            Bucket=os.environ["BUCKET_NAME"],
            Key=request["artifact_key"],
            Body=payload,
            ContentType="image/png",
        )

    result = {
        "key": request["artifact_key"],
        "sha256": digest,
        "content_type": "image/png",
        "width": width,
        "height": height,
    }
    encoded = json.dumps(result, separators=(",", ":")).encode()
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise ValueError("artifact result exceeds 4096 bytes")
    Path(args.output).write_bytes(encoded)


if __name__ == "__main__":
    main()
