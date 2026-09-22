from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    request = json.loads(Path(args.input).read_text())
    width, height = SIZES[request["wh_ratio"]]
    model_dir = Path(os.environ["MODEL_PATH"])
    with tempfile.TemporaryDirectory() as output_dir:
        image_path = Path(output_dir) / "image.png"
        subprocess.run(
            [
                "/sd.cpp/bin/sd-cli",
                "--diffusion-model",
                str(model_dir / "qwen_image_2.1-Q4_K.gguf"),
                "--vae",
                str(model_dir / "vae/qwen_image_2.1_vae_bf16.safetensors"),
                "--llm",
                str(model_dir / "Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
                "-p",
                request["prompt"],
                "--cfg-scale",
                "6",
                "--sampling-method",
                "euler",
                "--steps",
                str(request.get("steps", 40)),
                "--seed",
                str(request.get("seed", 42)),
                "--backend",
                "diffusion=cuda0,te=cpu,vae=cuda0",
                "--params-backend",
                "diffusion=disk,te=cpu,vae=cuda0",
                "--max-vram",
                "cuda0=6",
                "--diffusion-fa",
                "-W",
                str(width),
                "-H",
                str(height),
                "-o",
                str(image_path),
            ],
            check=True,
        )
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
