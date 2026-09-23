import json

import httpx
import pytest
from fastapi.testclient import TestClient

from image_gateway.api import create_app
from image_gateway.config import Settings

from .conftest import FakeCluster


class ProxyingCluster(FakeCluster):
    def create_request(self, namespace, body):
        result = super().create_request(namespace, body)
        status = {"phase": "Proxying", "serverUrl": "http://10.0.0.9:8080"}
        result["status"] = status
        self.requests[(namespace, body["metadata"]["name"])] = result
        return result


def test_health_probes_and_custom_job_apis_are_absent():
    client = TestClient(create_app(cluster=FakeCluster(), settings=Settings(namespace="images")))

    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready"}
    for path, method in (
        ("/v1/images/jobs", "post"),
        ("/v1/images/jobs/example", "get"),
        ("/v1/images/jobs/example/start", "post"),
        ("/v1/images/jobs/example/artifact", "get"),
        ("/docs", "get"),
        ("/openapi.json", "get"),
    ):
        assert getattr(client, method)(path).status_code == 404


def test_openai_generations_preserves_upstream_contract_and_prompt(monkeypatch):
    cluster = ProxyingCluster()
    forwarded = {}

    async def handle(request):
        forwarded["url"] = str(request.url)
        forwarded["body"] = await request.aread()
        forwarded["authorization"] = request.headers.get("authorization")
        forwarded["content_length"] = request.headers.get("content-length")
        return httpx.Response(
            201,
            headers={"content-type": "application/json", "x-sd-server": "true"},
            content=b'{"created":123,"data":[{"b64_json":"upstream"}]}',
        )

    monkeypatch.setattr(
        "image_gateway.api._sd_client",
        lambda timeout: httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    client = TestClient(
        create_app(
            cluster=cluster,
            settings=Settings(
                namespace="images",
                llm_gateway_api_key="secret",
            ),
        )
    )

    response = client.post(
        "/v1/images/generations",
        headers={"Authorization": "Bearer secret"},
        json={
            "model": "image-model",
            "prompt": "a fox in a forest",
            "size": "1792x1024",
            "n": 2,
            "response_format": "b64_json",
            "extra_upstream_field": {"preserve": True},
        },
    )

    assert response.status_code == 201
    assert response.content == b'{"created":123,"data":[{"b64_json":"upstream"}]}'
    assert response.headers["x-sd-server"] == "true"
    assert forwarded["url"] == "http://10.0.0.9:8080/v1/images/generations"
    assert forwarded["authorization"] is None
    assert forwarded["content_length"] == str(len(forwarded["body"]))
    assert json.loads(forwarded["body"]) == {
        "model": "image-model",
        "prompt": "a fox in a forest",
        "size": "1792x1024",
        "n": 2,
        "response_format": "b64_json",
        "extra_upstream_field": {"preserve": True},
    }
    generated = next(iter(cluster.requests.values()))
    assert generated["spec"]["operation"] == "openai"
    assert generated["spec"]["suspend"] is False
    assert not any(
        key.startswith("argocd.argoproj.io/")
        for key in generated["metadata"].get("annotations", {})
    )
    assert "prompt" not in generated["spec"]


def test_openai_edits_streams_multipart_body_unchanged(monkeypatch):
    cluster = ProxyingCluster()
    multipart_body = (
        b"--image-boundary\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\n"
        b"a cat\r\n--image-boundary--\r\n"
    )
    forwarded = {}

    async def handle(request):
        forwarded["body"] = await request.aread()
        forwarded["content_type"] = request.headers["content-type"]
        return httpx.Response(200, headers={"content-type": "image/png"}, content=b"png")

    monkeypatch.setattr(
        "image_gateway.api._sd_client",
        lambda timeout: httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    client = TestClient(
        create_app(
            cluster=cluster,
            settings=Settings(
                namespace="images",
                llm_gateway_api_key="secret",
            ),
        )
    )

    response = client.post(
        "/v1/images/edits",
        headers={
            "Authorization": "Bearer secret",
            "Content-Type": "multipart/form-data; boundary=image-boundary",
        },
        content=multipart_body,
    )

    assert response.status_code == 200
    assert response.content == b"png"
    assert response.headers["content-type"] == "image/png"
    assert forwarded["body"] == multipart_body
    assert forwarded["content_type"] == "multipart/form-data; boundary=image-boundary"
    generated = next(iter(cluster.requests.values()))
    assert generated["spec"]["operation"] == "openai"
    assert "optimizePrompt" not in generated["spec"]
    assert "prompt" not in generated["spec"]


def test_generation_validation_is_delegated_to_stable_diffusion_server(monkeypatch):
    cluster = ProxyingCluster()
    raw_body = b"{invalid json"
    forwarded = []

    async def handle(request):
        forwarded.append(await request.aread())
        return httpx.Response(
            400,
            headers={"content-type": "application/json"},
            content=b'{"error":{"message":"invalid request"}}',
        )

    monkeypatch.setattr(
        "image_gateway.api._sd_client",
        lambda timeout: httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    client = TestClient(
        create_app(
            cluster=cluster,
            settings=Settings(namespace="images", llm_gateway_api_key="secret"),
        )
    )

    response = client.post(
        "/v1/images/generations",
        headers={"Authorization": "Bearer secret"},
        content=raw_body,
    )

    assert response.status_code == 400
    assert response.content == b'{"error":{"message":"invalid request"}}'
    assert forwarded == [raw_body]


@pytest.mark.parametrize(
    ("headers", "status", "code"),
    [
        ({}, 401, "invalid_api_key"),
        ({"Authorization": "Bearer wrong"}, 401, "invalid_api_key"),
    ],
)
def test_openai_authentication_errors(headers, status, code):
    client = TestClient(
        create_app(
            cluster=ProxyingCluster(),
            settings=Settings(namespace="images", llm_gateway_api_key="secret"),
        )
    )

    response = client.post("/v1/images/generations", headers=headers, json={"prompt": "a cat"})

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
