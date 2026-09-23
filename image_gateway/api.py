from __future__ import annotations

import base64
import hmac
import time
from typing import Any
from urllib.request import Request, urlopen
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi import Request as FastAPIRequest
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .argocd import tracking_annotations
from .cluster import ClusterError, ConflictError, KubernetesCluster, NotFoundError
from .config import Settings
from .models import (
    OPENAI_SIZE_RATIOS,
    ArtifactResponse,
    ImageJobCreate,
    ImageJobResponse,
    OpenAIImageGenerationRequest,
    ProbeResponse,
)
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


def _openai_error(
    message: str,
    *,
    error_type: str,
    code: str,
    status_code: int,
    param: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
    )


def _download_image(url: str) -> bytes:
    request = Request(url, method="GET")
    with urlopen(request, timeout=120) as response:
        payload = response.read(256 * 1024 * 1024 + 1)
    if len(payload) > 256 * 1024 * 1024:
        raise ValueError("generated image exceeds the response size limit")
    return payload


def create_app(*, cluster: Any | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    cluster = cluster or KubernetesCluster()
    app = FastAPI(title="image-generation-gateway", version="0.1.0")

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        request: FastAPIRequest, exc: RequestValidationError
    ) -> JSONResponse:
        if request.url.path != "/v1/images/generations":
            return await request_validation_exception_handler(request, exc)
        first_error = exc.errors()[0] if exc.errors() else {}
        location = first_error.get("loc", ())
        param = str(location[-1]) if location and location[-1] != "body" else None
        message = first_error.get("msg", "invalid request")
        return _openai_error(
            message,
            error_type="invalid_request_error",
            code="invalid_request",
            status_code=400,
            param=param,
        )

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

    @app.post("/v1/images/generations")
    def generate_image(
        request: OpenAIImageGenerationRequest,
        http_request: FastAPIRequest,
    ) -> JSONResponse:
        if not settings.llm_gateway_api_key:
            return _openai_error(
                "image generation authentication is not configured",
                error_type="server_error",
                code="authentication_not_configured",
                status_code=503,
            )
        authorization = http_request.headers.get("authorization", "")
        scheme, separator, token = authorization.partition(" ")
        if (
            not separator
            or scheme.lower() != "bearer"
            or not hmac.compare_digest(token, settings.llm_gateway_api_key)
        ):
            return _openai_error(
                "invalid API key",
                error_type="authentication_error",
                code="invalid_api_key",
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        request_id = f"hermes-{uuid4()}"
        job = ImageJobCreate(
            idempotencyKey=request_id,
            prompt=request.prompt,
            aspectRatio=OPENAI_SIZE_RATIOS[request.size],
        )
        name = request_name(request_id)
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
            "spec": {**job.model_dump(by_alias=True), "suspend": False},
        }

        try:
            cluster.create_request(settings.namespace, body)
        except ClusterError:
            return _openai_error(
                "image generation request could not be created",
                error_type="server_error",
                code="request_creation_failed",
                status_code=503,
            )

        deadline = time.monotonic() + settings.image_generation_timeout_seconds
        while True:
            try:
                cr = cluster.get_request(settings.namespace, name)
            except ClusterError:
                return _openai_error(
                    "image generation request could not be read",
                    error_type="server_error",
                    code="request_read_failed",
                    status_code=503,
                )
            status = cr.get("status", {})
            phase = status.get("phase", "Pending")
            if phase == "Failed":
                return _openai_error(
                    "image generation failed",
                    error_type="server_error",
                    code="image_generation_failed",
                    status_code=500,
                )
            if phase == "Succeeded":
                artifact = status.get("artifact")
                if not artifact:
                    return _openai_error(
                        "image generation completed without an artifact",
                        error_type="server_error",
                        code="artifact_missing",
                        status_code=500,
                    )
                try:
                    artifact_url = cluster.artifact_url(settings.artifact_endpoint, artifact)
                    image_data: dict[str, str]
                    if request.response_format == "url":
                        image_data = {"url": artifact_url}
                    else:
                        encoded = base64.b64encode(_download_image(artifact_url)).decode("ascii")
                        image_data = {"b64_json": encoded}
                except Exception:
                    return _openai_error(
                        "generated image could not be retrieved",
                        error_type="server_error",
                        code="artifact_retrieval_failed",
                        status_code=500,
                    )
                revised_prompt = status.get("optimizer", {}).get("rewrittenPrompt")
                if revised_prompt:
                    image_data["revised_prompt"] = revised_prompt
                return JSONResponse(
                    content={"created": int(time.time()), "data": [image_data]}
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return _openai_error(
                    "image generation timed out",
                    error_type="server_error",
                    code="image_generation_timeout",
                    status_code=504,
                )
            time.sleep(min(settings.api_poll_interval_seconds, remaining))

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
