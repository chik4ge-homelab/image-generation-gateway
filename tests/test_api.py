import pytest
from fastapi.testclient import TestClient

from image_gateway.api import create_app
from image_gateway.config import Settings
from image_gateway.models import ImageJobCreate
from image_gateway.naming import request_name

from .conftest import FakeCluster


class SucceededCluster(FakeCluster):
    def create_request(self, namespace, body):
        result = super().create_request(namespace, body)
        result["status"] = {
            "phase": "Succeeded",
            "optimizer": {"rewrittenPrompt": "a carefully composed fox", "whRatio": "16:9"},
            "artifact": {
                "key": "image-generation/generated.png",
                "sha256": "0" * 64,
                "contentType": "image/png",
                "width": 1344,
                "height": 768,
            },
        }
        self.requests[(namespace, body["metadata"]["name"])] = result
        return result


def routes(app):
    return {route.path: route.endpoint for route in app.routes if hasattr(route, "path")}


def test_api_idempotency_and_start():
    cluster = FakeCluster()
    settings = Settings(namespace="images", argocd_application_name="llm-gateway")
    endpoints = routes(create_app(cluster=cluster, settings=settings))
    payload = ImageJobCreate(idempotencyKey="once", prompt="a small cat")
    first = endpoints["/v1/images/jobs"](payload)
    second = endpoints["/v1/images/jobs"](payload)
    assert first["job_id"] == second["job_id"] == request_name("once")
    assert len(cluster.requests) == 1
    assert cluster.requests[("images", request_name("once"))]["spec"]["suspend"] is True
    assert cluster.requests[("images", request_name("once"))]["metadata"]["annotations"] == {
        "argocd.argoproj.io/tracking-id": (
            f"llm-gateway:homelab.chik4ge.me/ImageGenerationRequest:images/"
            f"{request_name('once')}"
        ),
        "argocd.argoproj.io/compare-options": "IgnoreExtraneous",
        "argocd.argoproj.io/sync-options": "Prune=false",
    }

    with pytest.raises(Exception) as conflict:
        endpoints["/v1/images/jobs"](
            ImageJobCreate(idempotencyKey="once", prompt="a different prompt")
        )
    assert conflict.value.status_code == 409

    started = endpoints["/v1/images/jobs/{job_id}/start"](request_name("once"))
    assert started["spec"]["suspend"] is False


def test_api_health_and_artifact_not_ready():
    cluster = FakeCluster()
    endpoints = routes(create_app(cluster=cluster, settings=Settings(namespace="images")))
    assert endpoints["/healthz"]() == {"status": "ok"}
    assert endpoints["/readyz"]() == {"status": "ready"}
    with pytest.raises(Exception) as missing:
        endpoints["/v1/images/jobs/{job_id}/artifact"]("missing")
    assert missing.value.status_code == 404


def test_api_response_models_serialize_crd_field_names():
    cluster = FakeCluster()
    client = TestClient(create_app(cluster=cluster, settings=Settings(namespace="images")))

    response = client.post(
        "/v1/images/jobs",
        json={"idempotencyKey": "response-contract", "prompt": "a fox"},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["spec"] == {
        "idempotencyKey": "response-contract",
        "prompt": "a fox",
        "aspectRatio": "1:1",
        "steps": 40,
        "seed": 42,
        "suspend": True,
    }
    assert body["phase"] == "Pending"


def test_artifact_endpoint_returns_download_url():
    cluster = FakeCluster()
    cr = cluster.create_request(
        "images",
        {
            "metadata": {"name": "igr-artifact"},
            "spec": {
                "idempotencyKey": "artifact",
                "prompt": "a fox",
                "suspend": False,
            },
        },
    )
    cluster.patch_request_status(
        "images",
        cr["metadata"]["name"],
        {
            "phase": "Succeeded",
            "artifact": {
                "key": "image-generation/igr-artifact.png",
                "sha256": "0" * 64,
                "contentType": "image/png",
                "width": 1024,
                "height": 1024,
            },
        },
    )
    client = TestClient(
        create_app(
            cluster=cluster,
            settings=Settings(namespace="images", artifact_endpoint="http://objects"),
        )
    )
    response = client.get("/v1/images/jobs/igr-artifact/artifact")
    assert response.status_code == 200
    assert response.json()["url"] == (
        "http://objects/image-generation/igr-artifact.png"
    )


def test_openai_images_generations_uses_authenticated_standard_contract():
    cluster = SucceededCluster()
    client = TestClient(
        create_app(
            cluster=cluster,
            settings=Settings(
                namespace="images",
                argocd_application_name="llm-gateway",
                artifact_endpoint="https://objects.example",
                llm_gateway_api_key="secret",
            ),
        )
    )

    response = client.post(
        "/v1/images/generations",
        headers={"Authorization": "Bearer secret"},
        json={
            "model": "dall-e-3",
            "prompt": "a fox in a forest",
            "size": "1792x1024",
            "response_format": "url",
            "n": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["created"], int)
    assert body["data"] == [
        {
            "url": "https://objects.example/image-generation/generated.png",
            "revised_prompt": "a carefully composed fox",
        }
    ]
    generated = next(iter(cluster.requests.values()))
    assert generated["spec"]["aspectRatio"] == "16:9"
    assert generated["spec"]["suspend"] is False


def test_openai_images_generations_supports_base64_response(monkeypatch):
    cluster = SucceededCluster()
    client = TestClient(
        create_app(
            cluster=cluster,
            settings=Settings(
                namespace="images", llm_gateway_api_key="secret", artifact_endpoint="https://objects"
            ),
        )
    )
    monkeypatch.setattr("image_gateway.api._download_image", lambda url: b"png-bytes")

    response = client.post(
        "/v1/images/generations",
        headers={"Authorization": "Bearer secret"},
        json={"prompt": "a cat", "response_format": "b64_json"},
    )

    assert response.status_code == 200
    assert response.json()["data"][0]["b64_json"] == "cG5nLWJ5dGVz"


@pytest.mark.parametrize(
    ("headers", "payload", "status", "code"),
    [
        ({}, {"prompt": "a cat"}, 401, "invalid_api_key"),
        (
            {"Authorization": "Bearer secret"},
            {"prompt": "a cat", "size": "auto"},
            400,
            "invalid_request",
        ),
        (
            {"Authorization": "Bearer secret"},
            {"prompt": "a cat", "n": 2},
            400,
            "invalid_request",
        ),
    ],
)
def test_openai_images_generations_uses_openai_error_shape(headers, payload, status, code):
    client = TestClient(
        create_app(
            cluster=SucceededCluster(),
            settings=Settings(namespace="images", llm_gateway_api_key="secret"),
        )
    )

    response = client.post("/v1/images/generations", headers=headers, json=payload)

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
