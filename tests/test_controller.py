from image_gateway.config import Settings
from image_gateway.controller import Controller
from image_gateway.naming import diffuser_job_name, optimizer_job_name

from .conftest import FakeCluster, artifact_output, make_cr


def controller_settings():
    return Settings(
        namespace="images",
        optimizer_image="optimizer:test",
        diffuser_image="diffuser:test",
        optimizer_model_path="/models/optimizer.gguf",
        diffuser_model_path="/models/diffuser.safetensors",
        object_bucket_secret_name="object-bucket",
        artifact_endpoint="https://objects.example.test",
    )


def test_fifo_claim_and_full_phase_transition():
    cluster = FakeCluster()
    cr = make_cr()
    cr["metadata"]["namespace"] = "images"
    cluster.requests[("images", cr["metadata"]["name"])] = cr
    controller = Controller(cluster, controller_settings())

    assert controller.run_once() is True
    assert cluster.get_request("images", cr["metadata"]["name"])["status"]["phase"] == "Optimizing"

    assert controller.run_once() is True
    optimizer_name = optimizer_job_name(cr["metadata"]["uid"])
    cluster.jobs[("images", optimizer_name)]["status"] = {"succeeded": 1}
    cluster.outputs[("images", optimizer_name)] = '{"rewritten_prompt":"a cat","wh_ratio":"1:1"}'

    assert controller.run_once() is True
    assert cluster.get_request("images", cr["metadata"]["name"])["status"]["phase"] == "Generating"

    assert controller.run_once() is True
    diffuser_name = diffuser_job_name(cr["metadata"]["uid"])
    cluster.jobs[("images", diffuser_name)]["status"] = {"succeeded": 1}
    cluster.outputs[("images", diffuser_name)] = artifact_output()

    assert controller.run_once() is True
    assert (
        cluster.get_request("images", cr["metadata"]["name"])["status"]["phase"] == "RestoringText"
    )
    assert controller.run_once() is True
    result = cluster.get_request("images", cr["metadata"]["name"])
    assert result["status"]["phase"] == "Succeeded"
    assert result["status"]["artifact"]["contentType"] == "image/png"
    assert cluster.scales[0][2] == 0
    assert cluster.scales[-1][2] == 1


def test_suspended_request_does_not_claim():
    cluster = FakeCluster()
    cr = make_cr()
    cr["spec"]["suspend"] = True
    cluster.requests[("images", cr["metadata"]["name"])] = cr
    assert Controller(cluster, controller_settings()).run_once() is False
    assert "status" not in cluster.get_request("images", cr["metadata"]["name"])


def test_fifo_claims_only_the_oldest_ready_request():
    cluster = FakeCluster()
    oldest = make_cr("igr-oldest")
    oldest["metadata"]["creationTimestamp"] = "2026-01-01T00:00:00Z"
    newest = make_cr("igr-newest")
    newest["metadata"]["uid"] = "87654321-4321-4321-4321-cba987654321"
    newest["metadata"]["creationTimestamp"] = "2026-01-01T00:01:00Z"
    cluster.requests[("images", newest["metadata"]["name"])] = newest
    cluster.requests[("images", oldest["metadata"]["name"])] = oldest

    assert Controller(cluster, controller_settings()).run_once() is True

    assert cluster.get_request("images", oldest["metadata"]["name"])["status"]["phase"] == (
        "Optimizing"
    )
    assert "status" not in cluster.get_request("images", newest["metadata"]["name"])


def test_claim_rejects_a_stale_resource_version():
    cluster = FakeCluster()
    stale = make_cr()
    current = make_cr()
    current["metadata"]["resourceVersion"] = "2"
    cluster.requests[("images", current["metadata"]["name"])] = current

    assert Controller(cluster, controller_settings())._claim(stale) is False
    assert "status" not in cluster.get_request("images", current["metadata"]["name"])


def test_active_request_blocks_an_older_pending_request():
    cluster = FakeCluster()
    pending = make_cr("igr-pending")
    pending["metadata"]["creationTimestamp"] = "2026-01-01T00:00:00Z"
    active = make_cr("igr-active")
    active["metadata"]["uid"] = "87654321-4321-4321-4321-cba987654321"
    active["metadata"]["creationTimestamp"] = "2026-01-01T00:01:00Z"
    active["status"] = {
        "phase": "Generating",
        "optimizer": {"rewrittenPrompt": "a cat", "whRatio": "1:1"},
    }
    cluster.requests[("images", pending["metadata"]["name"])] = pending
    cluster.requests[("images", active["metadata"]["name"])] = active

    assert Controller(cluster, controller_settings()).run_once() is True

    assert "status" not in cluster.get_request("images", pending["metadata"]["name"])
    assert cluster.get_request("images", active["metadata"]["name"])["status"]["phase"] == (
        "Generating"
    )


def test_llm_scale_down_timeout_fails_request_and_restores_llm():
    cluster = FakeCluster()
    cr = make_cr()
    cr["status"] = {
        "phase": "StoppingText",
        "startedAt": "2020-01-01T00:00:00Z",
    }
    cluster.pods_gone = False
    cluster.requests[("images", cr["metadata"]["name"])] = cr
    settings = controller_settings()
    settings = settings.__class__(**{**settings.__dict__, "llm_stop_timeout_seconds": 0})

    assert Controller(cluster, settings).run_once() is True
    result = cluster.get_request("images", cr["metadata"]["name"])
    assert result["status"]["phase"] == "Failed"
    assert result["status"]["failure"]["reason"] == "LlmScaleDownTimeout"
    assert cluster.scales[-1][2] == 1
