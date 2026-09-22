from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from .argocd import tracking_annotations
from .cluster import ClusterError, ConflictError, KubernetesCluster, NotFoundError
from .config import Settings
from .models import ArtifactResponse, ImageJobCreate, ImageJobResponse, ProbeResponse
from .naming import request_name


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="image generation request not found")


def _job_response(cr: dict[str, Any]) -> dict[str, Any]:
    metadata = cr.get("metadata", {})
    status = cr.get("status", {})
    spec = cr.get("spec", {})
    response: dict[str, Any] = {
        "job_id": metadata.get("name"),
        "name": metadata.get("name"),
        "phase": status.get("phase", "Pending"),
        "spec": spec,
    }
    for key in ("optimizer", "artifact", "failure", "conditions"):
        if key in status:
            response[key] = status[key]
    return response


def create_app(*, cluster: Any | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    cluster = cluster or KubernetesCluster()
    app = FastAPI(title="image-generation-gateway", version="0.1.0")

    @app.get("/healthz", response_model=ProbeResponse)
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", response_model=ProbeResponse)
    def readyz() -> dict[str, str]:
        try:
            ready = bool(cluster.ready())
        except Exception as exc:
            raise HTTPException(status_code=503, detail="kubernetes client is not ready") from exc
        if not ready:
            raise HTTPException(status_code=503, detail="kubernetes client is not ready")
        return {"status": "ready"}

    @app.post("/v1/images/jobs", status_code=202, response_model=ImageJobResponse)
    def create_job(request: ImageJobCreate) -> dict[str, Any]:
        name = request_name(request.idempotency_key)
        body = {
            "apiVersion": "homelab.chik4ge.me/v1alpha1",
            "kind": "ImageGenerationRequest",
            "metadata": {
                "name": name,
                "namespace": settings.namespace,
                "labels": {"app.kubernetes.io/part-of": "image-generation"},
                "annotations": tracking_annotations(
                    settings.argocd_application_name,
                    group="homelab.chik4ge.me",
                    kind="ImageGenerationRequest",
                    namespace=settings.namespace,
                    name=name,
                ),
            },
            "spec": {
                **request.model_dump(by_alias=True),
                "suspend": True,
            },
        }
        try:
            cr = cluster.create_request(settings.namespace, body)
            return _job_response(cr)
        except ConflictError:
            try:
                existing = cluster.get_request(settings.namespace, name)
                existing_spec = existing.get("spec", {})
                requested_spec = body["spec"]
                comparable_fields = (
                    "idempotencyKey",
                    "prompt",
                    "aspectRatio",
                    "steps",
                    "seed",
                )
                if any(
                    existing_spec.get(field) != requested_spec.get(field)
                    for field in comparable_fields
                ):
                    raise HTTPException(
                        status_code=409,
                        detail="idempotency key is already used for a different request",
                    )
                return _job_response(existing)
            except NotFoundError as exc:
                raise HTTPException(status_code=409, detail="request creation conflicted") from exc
        except ClusterError as exc:
            raise HTTPException(status_code=503, detail="kubernetes request failed") from exc

    @app.get("/v1/images/jobs/{job_id}", response_model=ImageJobResponse)
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            return _job_response(cluster.get_request(settings.namespace, job_id))
        except NotFoundError as exc:
            raise _not_found() from exc
        except ClusterError as exc:
            raise HTTPException(status_code=503, detail="kubernetes request failed") from exc

    @app.post("/v1/images/jobs/{job_id}/start", response_model=ImageJobResponse)
    def start_job(job_id: str) -> dict[str, Any]:
        try:
            current = cluster.get_request(settings.namespace, job_id)
            if current.get("status", {}).get("phase") in {"Succeeded", "Failed"}:
                raise HTTPException(status_code=409, detail="job is terminal")
            cr = cluster.patch_request(settings.namespace, job_id, {"spec": {"suspend": False}})
            return _job_response(cr)
        except NotFoundError as exc:
            raise _not_found() from exc
        except HTTPException:
            raise
        except ClusterError as exc:
            raise HTTPException(status_code=503, detail="kubernetes request failed") from exc

    @app.get("/v1/images/jobs/{job_id}/artifact", response_model=ArtifactResponse)
    def get_artifact(job_id: str) -> dict[str, Any]:
        try:
            cr = cluster.get_request(settings.namespace, job_id)
        except NotFoundError as exc:
            raise _not_found() from exc
        except ClusterError as exc:
            raise HTTPException(status_code=503, detail="kubernetes request failed") from exc
        artifact = cr.get("status", {}).get("artifact")
        if cr.get("status", {}).get("phase") != "Succeeded" or not artifact:
            raise HTTPException(status_code=404, detail="artifact is not ready")
        result = {"job_id": job_id, "artifact": artifact}
        if settings.artifact_endpoint and artifact.get("key"):
            result["url"] = cluster.artifact_url(settings.artifact_endpoint, artifact)
        return result

    return app
