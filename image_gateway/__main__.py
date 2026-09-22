from __future__ import annotations

import argparse
import logging

from .api import create_app
from .cluster import KubernetesCluster
from .config import Settings
from .controller import Controller


def main() -> None:
    parser = argparse.ArgumentParser(prog="image-gateway")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("api", help="run the CPU-only HTTP API")
    subparsers.add_parser("controller", help="run the single-consumer GPU controller")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    settings = Settings.from_env()
    if args.mode == "api":
        import uvicorn

        uvicorn.run(create_app(settings=settings), host="0.0.0.0", port=settings.port)
    else:
        Controller(
            KubernetesCluster(llm_pod_label_selector=settings.llm_pod_label_selector), settings
        ).run_forever()
