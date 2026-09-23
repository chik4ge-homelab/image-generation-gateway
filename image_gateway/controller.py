from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any

from .cluster import ConflictError, NotFoundError
from .config import Settings
from .manifests import SERVER_CRONJOB_NAME, job_from_cronjob
from .models import TERMINAL_PHASES
from .naming import server_job_name

LOGGER = logging.getLogger(__name__)
FAILED_OPENAI_REQUEST_RETENTION_SECONDS = 60


class Controller:
    def __init__(self, cluster: Any, settings: Settings):
        self.cluster = cluster
        self.settings = settings

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")

    def run_once(self) -> bool:
        requests = sorted(
            self.cluster.list_requests(self.settings.namespace),
            key=lambda item: (
                item.get("metadata", {}).get("creationTimestamp", ""),
                item["metadata"]["name"],
            ),
        )
        deleted = self._cleanup_terminal_requests(requests)
        if deleted:
            requests = [
                item for item in requests if item["metadata"]["name"] not in deleted
            ]
        active = [
            cr
            for cr in requests
            if cr.get("status", {}).get("phase", "Pending")
            not in TERMINAL_PHASES | {"Pending"}
        ]
        if active:
            self.reconcile(active[0])
            return True

        for cr in requests:
            phase = cr.get("status", {}).get("phase", "Pending")
            if phase != "Pending":
                continue
            spec = cr.get("spec", {})
            if spec.get("suspend", True):
                continue
            if not self._claim(cr):
                continue
            cr = self.cluster.get_request(self.settings.namespace, cr["metadata"]["name"])
            self.reconcile(cr)
            return True
        return bool(deleted)

    def _cleanup_terminal_requests(self, requests: list[dict[str, Any]]) -> set[str]:
        active_server_jobs = {
            status["serverJobName"]
            for request in requests
            if (status := request.get("status", {})).get("phase")
            not in TERMINAL_PHASES | {"Pending"}
            and status.get("serverJobName")
        }
        deleted: set[str] = set()
        for request in requests:
            name = request["metadata"]["name"]
            status = request.get("status", {})
            phase = status.get("phase")
            if phase not in TERMINAL_PHASES:
                continue
            metadata = request.get("metadata", {})
            annotations = metadata.get("annotations", {})
            labels = metadata.get("labels", {})
            if (
                "argocd.argoproj.io/tracking-id" in annotations
                or "app.kubernetes.io/instance" in labels
            ):
                continue
            if status.get("serverJobName") in active_server_jobs:
                continue
            if phase == "Failed" and not self._failure_retention_elapsed(status):
                continue
            try:
                self.cluster.delete_request(self.settings.namespace, name)
                deleted.add(name)
            except Exception:
                LOGGER.exception("failed to delete terminal image request %s", name)
        return deleted

    @staticmethod
    def _failure_retention_elapsed(status: dict[str, Any]) -> bool:
        completed_at = status.get("completedAt")
        if not completed_at:
            return False
        try:
            completed = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return False
        age = datetime.now(UTC) - completed
        return age.total_seconds() >= FAILED_OPENAI_REQUEST_RETENTION_SECONDS

    def run_forever(self) -> None:
        while True:
            try:
                self.run_once()
            except Exception:
                LOGGER.exception("controller reconcile loop failed")
            time.sleep(self.settings.poll_interval_seconds)

    def _claim(self, cr: dict[str, Any]) -> bool:
        metadata = cr["metadata"]
        try:
            self.cluster.patch_request_status(
                self.settings.namespace,
                metadata["name"],
                {"phase": "StoppingText", "startedAt": self._now()},
                resource_version=metadata.get("resourceVersion"),
            )
            return True
        except ConflictError:
            return False

    def reconcile(self, cr: dict[str, Any]) -> None:
        phase = cr.get("status", {}).get("phase", "Pending")
        if phase == "StoppingText":
            self._stopping_text(cr)
        elif phase == "Prepared":
            self._prepared(cr)
        elif phase == "StartingServer":
            self._starting_server(cr)
        elif phase == "Proxying":
            self._proxying(cr)
        elif phase == "RestoringText":
            self._restoring_text(cr)

    def _stopping_text(self, cr: dict[str, Any]) -> None:
        try:
            self.cluster.llm_scale(
                self.settings.llm_namespace, self.settings.llm_deployment_name, 0
            )
            if self.cluster.llm_pods_gone(
                self.settings.llm_namespace, self.settings.llm_deployment_name
            ):
                self._patch_phase(cr, "Prepared")
            elif self._phase_timed_out(cr):
                self._fail(
                    cr,
                    "StoppingText",
                    "LlmScaleDownTimeout",
                    "llama-cpp pods did not terminate before the timeout",
                )
        except Exception as exc:
            self._fail(cr, "StoppingText", "LlmScaleDownFailed", str(exc))

    def _prepared(self, cr: dict[str, Any]) -> None:
        requests = sorted(
            self.cluster.list_requests(self.settings.namespace),
            key=lambda item: (
                item.get("metadata", {}).get("creationTimestamp", ""),
                item["metadata"]["name"],
            ),
        )
        queued = next(
            (
                item
                for item in requests
                if item.get("status", {}).get("phase", "Pending") == "Pending"
                and not item.get("spec", {}).get("suspend", True)
            ),
            None,
        )
        if queued:
            try:
                self.cluster.patch_request_status(
                    self.settings.namespace,
                    queued["metadata"]["name"],
                    {"phase": "Prepared", "startedAt": self._now()},
                    resource_version=queued.get("metadata", {}).get("resourceVersion"),
                )
            except ConflictError:
                return
            return
        self._patch_phase(cr, "StartingServer")

    def _starting_server(self, cr: dict[str, Any]) -> None:
        name = cr["metadata"]["name"]
        try:
            job_name = server_job_name(cr["metadata"]["uid"])
            job = self._ensure_template_job(cr, SERVER_CRONJOB_NAME, job_name)
            outcome = self._job_outcome(job)
            if outcome in {"failed", "succeeded"}:
                self._fail(
                    cr,
                    "StartingServer",
                    "StableDiffusionServerJobFailed",
                    "stable-diffusion.cpp server Job exited before serving requests",
                )
                return
            endpoint = self.cluster.job_endpoint(self.settings.namespace, job_name, 8080)
            if not endpoint:
                return
            self.cluster.patch_request_status(
                self.settings.namespace,
                name,
                {"phase": "Proxying", "serverJobName": job_name, "serverUrl": endpoint},
            )
        except Exception as exc:
            self._fail(cr, "StartingServer", "ServerStartupFailed", str(exc))

    def _proxying(self, cr: dict[str, Any]) -> None:
        status = cr.get("status", {})
        job_name = status.get("serverJobName")
        if not job_name:
            self._fail(
                cr, "Proxying", "ServerStateMissing", "stable-diffusion.cpp Job name is missing"
            )
            return
        try:
            job = self.cluster.get_job(self.settings.namespace, job_name)
            if self._job_outcome(job) in {"failed", "succeeded"}:
                self._fail(
                    cr,
                    "Proxying",
                    "StableDiffusionServerExited",
                    "stable-diffusion.cpp server exited while serving requests",
                )
            elif self._request_timed_out(cr):
                self._fail(
                    cr,
                    "Proxying",
                    "ImageGenerationTimeout",
                    "image generation request exceeded its timeout",
                )
        except NotFoundError:
            self._fail(
                cr, "Proxying", "ServerJobMissing", "stable-diffusion.cpp server Job disappeared"
            )
        except Exception as exc:
            self._fail(cr, "Proxying", "ServerStateReadFailed", str(exc))

    def _restoring_text(self, cr: dict[str, Any]) -> None:
        try:
            status = cr.get("status", {})
            server_job = status.get("serverJobName")
            if server_job:
                if self._shared_server_in_use(cr, server_job):
                    self._patch_phase(cr, "Succeeded")
                    return
                if not status.get("serverClosing"):
                    if self._advance_proxy_queue(cr):
                        return
                    self.cluster.patch_request_status(
                        self.settings.namespace,
                        cr["metadata"]["name"],
                        {"serverClosing": True},
                    )
                self.cluster.delete_job(self.settings.namespace, server_job)
                if not self.cluster.job_pods_gone(self.settings.namespace, server_job):
                    return
            elif self._advance_proxy_queue(cr):
                return
            self.cluster.llm_scale(
                self.settings.llm_namespace, self.settings.llm_deployment_name, 1
            )
            if self.cluster.wait_llm_ready(
                self.settings.llm_namespace,
                self.settings.llm_deployment_name,
                self.settings.llm_restore_timeout_seconds,
            ):
                self._patch_phase(cr, "Succeeded")
            else:
                self._fail(
                    cr, "RestoringText", "LlmReadinessTimeout", "llama-cpp did not become ready"
                )
        except Exception as exc:
            self._fail(cr, "RestoringText", "LlmRestoreFailed", str(exc))

    def _advance_proxy_queue(self, cr: dict[str, Any]) -> bool:
        status = cr.get("status", {})
        server_url = status.get("serverUrl")
        server_job = status.get("serverJobName")
        if not server_url or not server_job:
            return False
        requests = sorted(
            self.cluster.list_requests(self.settings.namespace),
            key=lambda item: (
                item.get("metadata", {}).get("creationTimestamp", ""),
                item["metadata"]["name"],
            ),
        )
        queued = next(
            (
                item
                for item in requests
                if item.get("status", {}).get("phase", "Pending") in {"Pending", "Prepared"}
                and not item.get("spec", {}).get("suspend", True)
            ),
            None,
        )
        if not queued:
            return False
        next_status: dict[str, Any] = {
            "phase": "Proxying",
            "startedAt": self._now(),
            "serverJobName": server_job,
            "serverUrl": server_url,
        }
        self.cluster.patch_request_status(
            self.settings.namespace, queued["metadata"]["name"], next_status
        )
        self.cluster.patch_request_status(
            self.settings.namespace,
            cr["metadata"]["name"],
            {"phase": "Succeeded", "completedAt": self._now()},
        )
        return True

    def _shared_server_in_use(self, cr: dict[str, Any], server_job: str) -> bool:
        return any(
            request["metadata"]["name"] != cr["metadata"]["name"]
            and request.get("status", {}).get("phase") == "Proxying"
            and request.get("status", {}).get("serverJobName") == server_job
            for request in self.cluster.list_requests(self.settings.namespace)
        )

    def _ensure_template_job(
        self, cr: dict[str, Any], cronjob_name: str, name: str
    ) -> dict[str, Any]:
        try:
            return self.cluster.get_job(self.settings.namespace, name)
        except NotFoundError:
            try:
                cronjob = self.cluster.get_cronjob(self.settings.namespace, cronjob_name)
                manifest = job_from_cronjob(cr, cronjob, name=name)
                return self.cluster.create_job(self.settings.namespace, manifest)
            except ConflictError:
                return self.cluster.get_job(self.settings.namespace, name)

    @staticmethod
    def _job_outcome(job: dict[str, Any]) -> str:
        status = job.get("status", {})
        if status.get("failed", 0) > 0:
            return "failed"
        for condition in status.get("conditions", []):
            if condition.get("type") == "Failed" and condition.get("status") == "True":
                return "failed"
        if status.get("succeeded", 0) > 0:
            return "succeeded"
        for condition in status.get("conditions", []):
            if condition.get("type") == "Complete" and condition.get("status") == "True":
                return "succeeded"
        return "running"

    def _patch_phase(self, cr: dict[str, Any], phase: str) -> None:
        status: dict[str, Any] = {"phase": phase}
        if phase in TERMINAL_PHASES:
            status["completedAt"] = self._now()
        self.cluster.patch_request_status(self.settings.namespace, cr["metadata"]["name"], status)

    def _phase_timed_out(self, cr: dict[str, Any]) -> bool:
        started_at = cr.get("status", {}).get("startedAt")
        if not started_at:
            return False
        try:
            started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        except ValueError:
            return False
        elapsed = (datetime.now(UTC) - started).total_seconds()
        return elapsed >= self.settings.llm_stop_timeout_seconds

    def _request_timed_out(self, cr: dict[str, Any]) -> bool:
        started_at = cr.get("status", {}).get("startedAt")
        if not started_at:
            return False
        try:
            started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        except ValueError:
            return False
        elapsed = (datetime.now(UTC) - started).total_seconds()
        return elapsed >= self.settings.image_generation_timeout_seconds

    def _fail(self, cr: dict[str, Any], stage: str, reason: str, message: str) -> None:
        conditions: list[dict[str, Any]] = []
        keep_llm_stopped = False
        try:
            server_job = cr.get("status", {}).get("serverJobName")
            if server_job:
                self.cluster.delete_job(self.settings.namespace, server_job)
                self.cluster.wait_job_pods_gone(self.settings.namespace, server_job, 120)
            keep_llm_stopped = any(
                request["metadata"]["name"] != cr["metadata"]["name"]
                and request.get("status", {}).get("phase") in {"Prepared", "StartingServer"}
                for request in self.cluster.list_requests(self.settings.namespace)
            )
            if not keep_llm_stopped:
                self.cluster.llm_scale(
                    self.settings.llm_namespace, self.settings.llm_deployment_name, 1
                )
                ready = self.cluster.wait_llm_ready(
                    self.settings.llm_namespace,
                    self.settings.llm_deployment_name,
                    self.settings.llm_restore_timeout_seconds,
                )
                if not ready:
                    conditions.append(
                        {
                            "type": "LlmRestorationFailed",
                            "status": "True",
                            "reason": "ReadinessTimeout",
                            "message": "llama-cpp did not become ready after failure",
                            "lastTransitionTime": self._now(),
                        }
                    )
        except Exception as exc:
            conditions.append(
                {
                    "type": "LlmRestorationFailed",
                    "status": "True",
                    "reason": "ScaleOrReadinessError",
                    "message": str(exc),
                    "lastTransitionTime": self._now(),
                }
            )
        status: dict[str, Any] = {
            "phase": "Failed",
            "completedAt": self._now(),
            "failure": {"stage": stage, "reason": reason, "message": message},
        }
        if conditions:
            status["conditions"] = conditions
        try:
            self.cluster.patch_request_status(
                self.settings.namespace, cr["metadata"]["name"], status
            )
        except Exception:
            LOGGER.exception("failed to write Failed status for %s", cr["metadata"]["name"])
