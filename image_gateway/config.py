from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    namespace: str = "default"
    optimizer_image: str = ""
    diffuser_image: str = ""
    optimizer_model_path: str = ""
    diffuser_model_path: str = ""
    object_bucket_secret_name: str = ""
    artifact_endpoint: str = ""
    artifact_prefix: str = "image-generation"
    model_pvc_name: str = "image-generation-models"
    job_service_account_name: str = "image-generation-job"
    llm_namespace: str = "llm-gateway"
    llm_deployment_name: str = "llama-cpp"
    llm_pod_label_selector: str = "app.kubernetes.io/name=llm-gateway"
    gpu_resource_name: str = "nvidia.com/gpu"
    gpu_count: int = 1
    gpu_runtime_class_name: str = "nvidia"
    job_active_deadline_seconds: int = 1800
    job_ttl_seconds: int = 3600
    llm_stop_timeout_seconds: int = 300
    llm_restore_timeout_seconds: int = 60
    poll_interval_seconds: float = 2.0
    port: int = 8080

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            namespace=os.getenv("IMAGE_NAMESPACE", cls.namespace),
            optimizer_image=os.getenv("OPTIMIZER_IMAGE", ""),
            diffuser_image=os.getenv("DIFFUSER_IMAGE", ""),
            optimizer_model_path=os.getenv("OPTIMIZER_MODEL_PATH", ""),
            diffuser_model_path=os.getenv("DIFFUSER_MODEL_PATH", ""),
            object_bucket_secret_name=os.getenv("OBJECT_BUCKET_SECRET_NAME", ""),
            artifact_endpoint=os.getenv("ARTIFACT_ENDPOINT", ""),
            artifact_prefix=os.getenv("ARTIFACT_PREFIX", cls.artifact_prefix),
            model_pvc_name=os.getenv("MODEL_PVC_NAME", cls.model_pvc_name),
            job_service_account_name=os.getenv(
                "JOB_SERVICE_ACCOUNT_NAME", cls.job_service_account_name
            ),
            llm_namespace=os.getenv("LLM_NAMESPACE", cls.llm_namespace),
            llm_deployment_name=os.getenv("LLM_DEPLOYMENT_NAME", cls.llm_deployment_name),
            llm_pod_label_selector=os.getenv(
                "LLM_POD_LABEL_SELECTOR", cls.llm_pod_label_selector
            ),
            gpu_resource_name=os.getenv("GPU_RESOURCE_NAME", cls.gpu_resource_name),
            gpu_count=int(os.getenv("GPU_COUNT", str(cls.gpu_count))),
            gpu_runtime_class_name=os.getenv("GPU_RUNTIME_CLASS_NAME", cls.gpu_runtime_class_name),
            job_active_deadline_seconds=int(
                os.getenv("JOB_ACTIVE_DEADLINE_SECONDS", str(cls.job_active_deadline_seconds))
            ),
            job_ttl_seconds=int(os.getenv("JOB_TTL_SECONDS", str(cls.job_ttl_seconds))),
            llm_stop_timeout_seconds=int(
                os.getenv("LLM_STOP_TIMEOUT_SECONDS", str(cls.llm_stop_timeout_seconds))
            ),
            llm_restore_timeout_seconds=int(
                os.getenv("LLM_RESTORE_TIMEOUT_SECONDS", str(cls.llm_restore_timeout_seconds))
            ),
            poll_interval_seconds=float(
                os.getenv("CONTROLLER_POLL_INTERVAL_SECONDS", str(cls.poll_interval_seconds))
            ),
            port=int(os.getenv("IMAGE_GATEWAY_PORT", str(cls.port))),
        )

    def validate_controller(self) -> None:
        missing = [
            name
            for name, value in (
                ("OPTIMIZER_IMAGE", self.optimizer_image),
                ("DIFFUSER_IMAGE", self.diffuser_image),
                ("OPTIMIZER_MODEL_PATH", self.optimizer_model_path),
                ("DIFFUSER_MODEL_PATH", self.diffuser_model_path),
                ("OBJECT_BUCKET_SECRET_NAME", self.object_bucket_secret_name),
                ("ARTIFACT_ENDPOINT", self.artifact_endpoint),
            )
            if not value
        ]
        if missing:
            raise ValueError("missing controller configuration: " + ", ".join(missing))
