from __future__ import annotations

import asyncio
import hmac
import time
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException
from fastapi import Request as FastAPIRequest
from fastapi.responses import JSONResponse, StreamingResponse

from .cluster import ClusterError, KubernetesCluster
from .config import Settings
from .models import ProbeResponse
from .naming import request_name


def _openai_error(
    message: str,
    *,
    error_type: str,
    code: str,
    status_code: int,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": code,
            }
        },
    )


def _forward_request_headers(request: FastAPIRequest) -> dict[str, str]:
    allowed = {
        "content-type",
        "content-encoding",
        "accept",
        "accept-encoding",
    }
    return {key: value for key, value in request.headers.items() if key.lower() in allowed}


def _response_headers(headers: httpx.Headers) -> dict[str, str]:
    hop_by_hop = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
    connection_tokens = {
        token.strip().lower()
        for token in headers.get("connection", "").split(",")
        if token.strip()
    }
    blocked = hop_by_hop | connection_tokens
    return {key: value for key, value in headers.items() if key.lower() not in blocked}


def _sd_client(timeout_seconds: int) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))


def create_app(*, cluster: Any | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    cluster = cluster or KubernetesCluster()
    app = FastAPI(
        title="image-generation-gateway",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/healthz", response_model=ProbeResponse)
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", response_model=ProbeResponse)
    def readyz() -> dict[str, str]:
        try:
            ready = bool(cluster.ready())
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Kubernetes client is not ready") from exc
        if not ready:
            raise HTTPException(status_code=503, detail="Kubernetes client is not ready")
        return {"status": "ready"}

    @app.post("/v1/images/generations")
    @app.post("/v1/images/edits")
    async def proxy_openai_images(http_request: FastAPIRequest) -> Any:
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

        is_generation = http_request.url.path.endswith("/generations")
        raw_body = await http_request.body() if is_generation else None
        request_id = f"openai-{uuid4()}"
        name = request_name(request_id)
        spec: dict[str, Any] = {
            "idempotencyKey": request_id,
            "operation": "openai",
            "suspend": False,
        }
        body = {
            "apiVersion": "homelab.chik4ge.me/v1alpha1",
            "kind": "ImageGenerationRequest",
            "metadata": {
                "name": name,
                "namespace": settings.namespace,
                "labels": {"app.kubernetes.io/part-of": "image-generation"},
            },
            "spec": spec,
        }
        try:
            await asyncio.to_thread(cluster.create_request, settings.namespace, body)
        except ClusterError:
            return _openai_error(
                "image generation request could not be created",
                error_type="server_error",
                code="request_creation_failed",
                status_code=503,
            )

        deadline = time.monotonic() + settings.image_generation_timeout_seconds
        try:
            while True:
                cr = await asyncio.to_thread(cluster.get_request, settings.namespace, name)
                status = cr.get("status", {})
                phase = status.get("phase", "Pending")
                if phase == "Proxying" and status.get("serverUrl"):
                    break
                if phase == "Failed":
                    failure = status.get("failure", {})
                    return _openai_error(
                        failure.get("message", "image generation failed"),
                        error_type="server_error",
                        code="image_generation_failed",
                        status_code=500,
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    await asyncio.to_thread(
                        cluster.patch_request_status,
                        settings.namespace,
                        name,
                        {"phase": "RestoringText"},
                    )
                    return _openai_error(
                        "image generation timed out",
                        error_type="server_error",
                        code="image_generation_timeout",
                        status_code=504,
                    )
                await asyncio.sleep(min(settings.api_poll_interval_seconds, remaining))
        except ClusterError:
            return _openai_error(
                "image generation request could not be read",
                error_type="server_error",
                code="request_read_failed",
                status_code=503,
            )

        content: bytes | Any = raw_body if raw_body is not None else http_request.stream()
        target = status["serverUrl"].rstrip("/") + http_request.url.path
        if http_request.url.query:
            target += "?" + http_request.url.query
        client = _sd_client(settings.image_generation_timeout_seconds)
        try:
            upstream_request = client.build_request(
                "POST",
                target,
                headers=_forward_request_headers(http_request),
                content=content,
            )
            upstream = await client.send(upstream_request, stream=True)
        except Exception:
            await client.aclose()
            await asyncio.to_thread(
                cluster.patch_request_status,
                settings.namespace,
                name,
                {"phase": "RestoringText"},
            )
            return _openai_error(
                "stable-diffusion.cpp request failed",
                error_type="server_error",
                code="image_generation_failed",
                status_code=502,
            )

        async def response_stream():
            try:
                if upstream.is_stream_consumed:
                    yield upstream.content
                else:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()
                try:
                    await asyncio.to_thread(
                        cluster.patch_request_status,
                        settings.namespace,
                        name,
                        {"phase": "RestoringText"},
                    )
                except Exception:
                    pass

        return StreamingResponse(
            response_stream(),
            status_code=upstream.status_code,
            headers=_response_headers(upstream.headers),
        )

    return app
