from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    namespace: str = "default"
    llm_namespace: str = "llm-gateway"
    llm_deployment_name: str = "llama-cpp"
    llm_pod_label_selector: str = "app.kubernetes.io/name=llm-gateway"
    llm_stop_timeout_seconds: int = 300
    llm_restore_timeout_seconds: int = 60
    poll_interval_seconds: float = 2.0
    api_poll_interval_seconds: float = 2.0
    image_generation_timeout_seconds: int = 3600
    request_heartbeat_interval_seconds: float = 15.0
    request_lease_timeout_seconds: int = 60
    llm_gateway_api_key: str = ""
    port: int = 8080

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            namespace=os.getenv("IMAGE_NAMESPACE", cls.namespace),
            llm_namespace=os.getenv("LLM_NAMESPACE", cls.llm_namespace),
            llm_deployment_name=os.getenv("LLM_DEPLOYMENT_NAME", cls.llm_deployment_name),
            llm_pod_label_selector=os.getenv("LLM_POD_LABEL_SELECTOR", cls.llm_pod_label_selector),
            llm_stop_timeout_seconds=int(
                os.getenv("LLM_STOP_TIMEOUT_SECONDS", str(cls.llm_stop_timeout_seconds))
            ),
            llm_restore_timeout_seconds=int(
                os.getenv("LLM_RESTORE_TIMEOUT_SECONDS", str(cls.llm_restore_timeout_seconds))
            ),
            poll_interval_seconds=float(
                os.getenv("CONTROLLER_POLL_INTERVAL_SECONDS", str(cls.poll_interval_seconds))
            ),
            api_poll_interval_seconds=float(
                os.getenv("API_POLL_INTERVAL_SECONDS", str(cls.api_poll_interval_seconds))
            ),
            image_generation_timeout_seconds=int(
                os.getenv(
                    "IMAGE_GENERATION_TIMEOUT_SECONDS", str(cls.image_generation_timeout_seconds)
                )
            ),
            request_heartbeat_interval_seconds=float(
                os.getenv(
                    "REQUEST_HEARTBEAT_INTERVAL_SECONDS",
                    str(cls.request_heartbeat_interval_seconds),
                )
            ),
            request_lease_timeout_seconds=int(
                os.getenv("REQUEST_LEASE_TIMEOUT_SECONDS", str(cls.request_lease_timeout_seconds))
            ),
            llm_gateway_api_key=os.getenv("LLM_GATEWAY_API_KEY", ""),
            port=int(os.getenv("IMAGE_GATEWAY_PORT", str(cls.port))),
        )
