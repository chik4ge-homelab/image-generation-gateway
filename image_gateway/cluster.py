from __future__ import annotations

import hashlib
import os
import time
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen


class ClusterError(RuntimeError):
    pass


class NotFoundError(ClusterError):
    pass


class ConflictError(ClusterError):
    pass


class KubernetesCluster:
    """Small replaceable adapter around the Kubernetes Python clients."""

    def __init__(
        self,
        *,
        custom_api: Any = None,
        batch_api: Any = None,
        core_api: Any = None,
        apps_api: Any = None,
        llm_pod_label_selector: str | None = None,
    ):
        from kubernetes import client, config

        if any(api is None for api in (custom_api, batch_api, core_api, apps_api)):
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
            custom_api = custom_api or client.CustomObjectsApi()
            batch_api = batch_api or client.BatchV1Api()
            core_api = core_api or client.CoreV1Api()
            apps_api = apps_api or client.AppsV1Api()
        self.custom = custom_api
        self.batch = batch_api
        self.core = core_api
        self.apps = apps_api
        self.llm_pod_label_selector = llm_pod_label_selector or os.getenv(
            "LLM_POD_LABEL_SELECTOR", "app.kubernetes.io/name=llm-gateway"
        )

    @staticmethod
    def _api_error(exc: Exception) -> ClusterError:
        status = getattr(exc, "status", None)
        if status == 404:
            return NotFoundError(str(exc))
        if status == 409:
            return ConflictError(str(exc))
        return ClusterError(str(exc))

    def ready(self) -> bool:
        return True

    def get_request(self, namespace: str, name: str) -> dict[str, Any]:
        try:
            return self.custom.get_namespaced_custom_object(
                "homelab.chik4ge.me", "v1alpha1", namespace, "imagegenerationrequests", name
            )
        except Exception as exc:
            raise self._api_error(exc) from exc

    def create_request(self, namespace: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.custom.create_namespaced_custom_object(
                "homelab.chik4ge.me", "v1alpha1", namespace, "imagegenerationrequests", body
            )
        except Exception as exc:
            raise self._api_error(exc) from exc

    def patch_request(self, namespace: str, name: str, patch: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.custom.patch_namespaced_custom_object(
                "homelab.chik4ge.me", "v1alpha1", namespace, "imagegenerationrequests", name, patch
            )
        except Exception as exc:
            raise self._api_error(exc) from exc

    def list_requests(self, namespace: str) -> list[dict[str, Any]]:
        try:
            result = self.custom.list_namespaced_custom_object(
                "homelab.chik4ge.me", "v1alpha1", namespace, "imagegenerationrequests"
            )
            return result.get("items", [])
        except Exception as exc:
            raise self._api_error(exc) from exc

    def delete_request(self, namespace: str, name: str) -> None:
        try:
            self.custom.delete_namespaced_custom_object(
                "homelab.chik4ge.me",
                "v1alpha1",
                namespace,
                "imagegenerationrequests",
                name,
                propagation_policy="Foreground",
            )
        except Exception as exc:
            error = self._api_error(exc)
            if isinstance(error, NotFoundError):
                return
            raise error from exc

    def patch_request_status(
        self, namespace: str, name: str, status: dict[str, Any], resource_version: str | None = None
    ) -> dict[str, Any]:
        patch: dict[str, Any] = {"status": status}
        if resource_version is not None:
            patch["metadata"] = {"resourceVersion": resource_version}
        try:
            method = getattr(self.custom, "patch_namespaced_custom_object_status", None)
            if method is None:
                method = self.custom.patch_namespaced_custom_object
            return method(
                "homelab.chik4ge.me", "v1alpha1", namespace, "imagegenerationrequests", name, patch
            )
        except Exception as exc:
            raise self._api_error(exc) from exc

    def get_configmap(self, namespace: str, name: str) -> dict[str, Any]:
        try:
            obj = self.core.read_namespaced_config_map(name, namespace)
            return self._serialize(obj)
        except Exception as exc:
            raise self._api_error(exc) from exc

    def create_configmap(self, namespace: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            obj = self.core.create_namespaced_config_map(namespace, body)
            return self._serialize(obj)
        except Exception as exc:
            raise self._api_error(exc) from exc

    def get_job(self, namespace: str, name: str) -> dict[str, Any]:
        try:
            obj = self.batch.read_namespaced_job(name, namespace)
            return self._serialize(obj)
        except Exception as exc:
            raise self._api_error(exc) from exc

    def get_cronjob(self, namespace: str, name: str) -> dict[str, Any]:
        try:
            obj = self.batch.read_namespaced_cron_job(name, namespace)
            return self._serialize(obj)
        except Exception as exc:
            raise self._api_error(exc) from exc

    def create_job(self, namespace: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            obj = self.batch.create_namespaced_job(namespace, body)
            return self._serialize(obj)
        except Exception as exc:
            raise self._api_error(exc) from exc

    def delete_job(self, namespace: str, name: str) -> None:
        try:
            self.batch.delete_namespaced_job(
                name, namespace, propagation_policy="Foreground"
            )
        except Exception as exc:
            error = self._api_error(exc)
            if isinstance(error, NotFoundError):
                return
            raise error from exc

    def job_pods_gone(self, namespace: str, job_name: str) -> bool:
        try:
            pods = self.core.list_namespaced_pod(
                namespace, label_selector=f"job-name={job_name}"
            )
            return not getattr(pods, "items", [])
        except Exception as exc:
            raise self._api_error(exc) from exc

    def wait_job_pods_gone(self, namespace: str, job_name: str, timeout_seconds: int) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.job_pods_gone(namespace, job_name):
                return True
            time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
        return self.job_pods_gone(namespace, job_name)

    def job_endpoint(self, namespace: str, job_name: str, port: int) -> str | None:
        try:
            pods = self.core.list_namespaced_pod(
                namespace, label_selector=f"job-name={job_name}"
            )
            for pod in getattr(pods, "items", []):
                pod_dict = self._serialize(pod)
                if pod_dict.get("status", {}).get("phase") != "Running":
                    continue
                conditions = pod_dict.get("status", {}).get("conditions", [])
                if not any(
                    condition.get("type") == "Ready" and condition.get("status") == "True"
                    for condition in conditions
                ):
                    continue
                pod_ip = pod_dict.get("status", {}).get("podIP")
                if pod_ip:
                    return f"http://{pod_ip}:{port}"
            return None
        except Exception as exc:
            raise self._api_error(exc) from exc

    def llm_scale(self, namespace: str, name: str, replicas: int) -> None:
        try:
            self.apps.patch_namespaced_deployment_scale(
                name, namespace, {"spec": {"replicas": replicas}}
            )
        except Exception as exc:
            raise self._api_error(exc) from exc

    def _llm_pods(self, namespace: str, name: str) -> list[dict[str, Any]]:
        try:
            scale = self._serialize(self.apps.read_namespaced_deployment_scale(name, namespace))
            selector = scale.get("status", {}).get("selector") or scale.get("spec", {}).get(
                "selector"
            )
            pods = self.core.list_namespaced_pod(
                namespace, label_selector=selector or self.llm_pod_label_selector
            )
            return [self._serialize(pod) for pod in getattr(pods, "items", [])]
        except Exception as exc:
            raise self._api_error(exc) from exc

    def llm_pods_gone(self, namespace: str, name: str) -> bool:
        return not self._llm_pods(namespace, name)

    def llm_ready(self, namespace: str, name: str) -> bool:
        pods = self._llm_pods(namespace, name)
        for pod in pods:
            if pod.get("status", {}).get("phase") != "Running":
                continue
            conditions = pod.get("status", {}).get("conditions", [])
            if any(
                condition.get("type") == "Ready" and condition.get("status") == "True"
                for condition in conditions
            ):
                return True
        return False

    def wait_llm_ready(self, namespace: str, name: str, timeout_seconds: int) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.llm_ready(namespace, name):
                return True
            time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
        return self.llm_ready(namespace, name)

    def read_job_output(self, namespace: str, job_name: str) -> str | None:
        try:
            pods = self.core.list_namespaced_pod(namespace, label_selector=f"job-name={job_name}")
            for pod in getattr(pods, "items", []):
                pod_dict = self._serialize(pod)
                statuses = pod_dict.get("status", {}).get("containerStatuses", [])
                for container in statuses:
                    terminated = container.get("state", {}).get("terminated")
                    if terminated and terminated.get("message"):
                        return terminated["message"]
            return None
        except Exception as exc:
            raise self._api_error(exc) from exc

    def verify_artifact(self, endpoint: str, artifact: dict[str, Any]) -> bool:
        if all(
            os.getenv(name)
            for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "BUCKET_NAME")
        ):
            return self._verify_s3_artifact(endpoint, artifact)
        return self._verify_http_artifact(endpoint, artifact)

    def artifact_url(self, endpoint: str, artifact: dict[str, Any]) -> str:
        if all(
            os.getenv(name)
            for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "BUCKET_NAME")
        ):
            endpoint_url = os.getenv("AWS_ENDPOINT_URL_S3") or os.getenv(
                "AWS_ENDPOINT_URL"
            ) or endpoint
            client = self._s3_client(endpoint_url)
            return client.generate_presigned_url(
                "get_object",
                Params={"Bucket": os.environ["BUCKET_NAME"], "Key": artifact["key"]},
                ExpiresIn=900,
            )
        return endpoint.rstrip("/") + "/" + quote(artifact["key"], safe="/")

    @staticmethod
    def _s3_client(endpoint_url: str):
        import boto3

        return boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=os.getenv("BUCKET_REGION") or "us-east-1",
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )

    @staticmethod
    def _verify_s3_artifact(endpoint: str, artifact: dict[str, Any]) -> bool:
        endpoint_url = os.getenv("AWS_ENDPOINT_URL_S3") or os.getenv("AWS_ENDPOINT_URL") or endpoint
        if not endpoint_url:
            host = os.getenv("BUCKET_HOST")
            port = os.getenv("BUCKET_PORT")
            if not host:
                return False
            endpoint_url = (
                host
                if host.startswith("http://") or host.startswith("https://")
                else f"http://{host}"
            )
            if port and "://" in endpoint_url and endpoint_url.rsplit(":", 1)[-1] != port:
                endpoint_url = f"{endpoint_url}:{port}"
        try:
            client = KubernetesCluster._s3_client(endpoint_url)
            response = client.get_object(Bucket=os.environ["BUCKET_NAME"], Key=artifact["key"])
            digest = hashlib.sha256()
            total = 0
            body = response["Body"]
            while chunk := body.read(1024 * 1024):
                total += len(chunk)
                if total > 256 * 1024 * 1024:
                    return False
                digest.update(chunk)
            return digest.hexdigest().lower() == artifact["sha256"].lower()
        except Exception:
            return False

    @staticmethod
    def _verify_http_artifact(endpoint: str, artifact: dict[str, Any]) -> bool:
        url = endpoint.rstrip("/") + "/" + quote(artifact["key"], safe="/")
        request = Request(url, method="GET")
        try:
            with urlopen(request, timeout=15) as response:
                if not 200 <= response.status < 300:
                    return False
                digest = hashlib.sha256()
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > 256 * 1024 * 1024:
                        return False
                    digest.update(chunk)
                return digest.hexdigest().lower() == artifact["sha256"].lower()
        except Exception:
            return False

    @staticmethod
    def _serialize(value: Any) -> dict[str, Any]:
        from kubernetes.client import ApiClient

        return ApiClient().sanitize_for_serialization(value)
