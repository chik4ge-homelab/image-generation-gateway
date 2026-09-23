from datetime import UTC, datetime

from image_gateway.config import Settings
from image_gateway.controller import Controller
from image_gateway.naming import server_job_name

from .conftest import FakeCluster, make_cr


def controller_settings():
    return Settings(namespace="images")


def test_fifo_claim_and_full_phase_transition():
    cluster = FakeCluster()
    cr = make_cr()
    cr["metadata"]["namespace"] = "images"
    cluster.requests[("images", cr["metadata"]["name"])] = cr
    job_name = server_job_name(cr["metadata"]["uid"])
    cluster.job_endpoints[("images", job_name)] = "http://10.1.2.3:8080"
    controller = Controller(cluster, controller_settings())

    assert controller.run_once() is True
    assert cluster.get_request("images", cr["metadata"]["name"])["status"]["phase"] == "Prepared"

    assert controller.run_once() is True
    assert (
        cluster.get_request("images", cr["metadata"]["name"])["status"]["phase"]
        == "StartingServer"
    )

    assert controller.run_once() is True
    assert cluster.get_request("images", cr["metadata"]["name"])["status"]["phase"] == "Proxying"
    cluster.patch_request_status("images", cr["metadata"]["name"], {"phase": "RestoringText"})

    assert controller.run_once() is True
    result = cluster.get_request("images", cr["metadata"]["name"])
    assert result["status"]["phase"] == "Succeeded"
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
        "Prepared"
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
    active_job_name = server_job_name(active["metadata"]["uid"])
    active["status"] = {"phase": "Proxying", "serverJobName": active_job_name}
    cluster.requests[("images", pending["metadata"]["name"])] = pending
    cluster.requests[("images", active["metadata"]["name"])] = active
    cluster.jobs[("images", active_job_name)] = {"status": {}}

    assert Controller(cluster, controller_settings()).run_once() is True

    assert "status" not in cluster.get_request("images", pending["metadata"]["name"])
    assert (
        cluster.get_request("images", active["metadata"]["name"])["status"]["phase"]
        == "Proxying"
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


def test_openai_queue_reuses_the_ready_stable_diffusion_server_job():
    cluster = FakeCluster()
    first = make_cr("igr-openai-first")
    first["metadata"]["uid"] = "12345678-1234-1234-1234-123456789abc"
    first["metadata"]["creationTimestamp"] = "2026-01-01T00:00:00Z"
    second = make_cr("igr-openai-second")
    second["metadata"]["uid"] = "87654321-4321-4321-4321-cba987654321"
    second["metadata"]["creationTimestamp"] = "2026-01-01T00:01:00Z"
    cluster.requests[("images", first["metadata"]["name"])] = first
    cluster.requests[("images", second["metadata"]["name"])] = second
    server_name = server_job_name(first["metadata"]["uid"])
    cluster.job_endpoints[("images", server_name)] = "http://10.1.2.3:8080"
    controller = Controller(cluster, controller_settings())

    assert controller.run_once()
    assert cluster.get_request("images", first["metadata"]["name"])["status"]["phase"] == (
        "Prepared"
    )
    assert controller.run_once()
    assert cluster.get_request("images", second["metadata"]["name"])["status"]["phase"] == (
        "Prepared"
    )
    assert controller.run_once()
    assert controller.run_once()
    first_status = cluster.get_request("images", first["metadata"]["name"])["status"]
    assert first_status["phase"] == "Proxying"
    assert first_status["serverJobName"] == server_name
    assert first_status["serverUrl"] == "http://10.1.2.3:8080"

    cluster.patch_request_status(
        "images", first["metadata"]["name"], {"phase": "RestoringText"}
    )
    assert controller.run_once()
    assert cluster.get_request("images", first["metadata"]["name"])["status"]["phase"] == (
        "Succeeded"
    )
    second_status = cluster.get_request("images", second["metadata"]["name"])["status"]
    assert second_status["phase"] == "Proxying"
    assert second_status["serverJobName"] == server_name
    assert second_status["serverUrl"] == "http://10.1.2.3:8080"
    assert ("images", server_name) not in cluster.deleted_jobs

    cluster.patch_request_status(
        "images", second["metadata"]["name"], {"phase": "RestoringText"}
    )
    assert controller.run_once()
    assert cluster.get_request("images", second["metadata"]["name"])["status"]["phase"] == (
        "Succeeded"
    )
    assert ("images", server_name) in cluster.deleted_jobs
    assert cluster.scales[-1][2] == 1


def test_fifo_requests_are_prepared_before_the_server_starts():
    cluster = FakeCluster()
    first = make_cr("igr-openai-first")
    first["metadata"]["uid"] = "12345678-1234-1234-1234-123456789abc"
    first["metadata"]["creationTimestamp"] = "2026-01-01T00:00:00Z"
    first["status"] = {"phase": "Prepared", "startedAt": "2026-01-01T00:00:00Z"}
    second = make_cr("igr-openai-second")
    second["metadata"]["uid"] = "87654321-4321-4321-4321-cba987654321"
    second["metadata"]["creationTimestamp"] = "2026-01-01T00:01:00Z"
    cluster.requests[("images", first["metadata"]["name"])] = first
    cluster.requests[("images", second["metadata"]["name"])] = second
    controller = Controller(cluster, controller_settings())

    assert controller.run_once()
    assert cluster.get_request("images", first["metadata"]["name"])["status"]["phase"] == (
        "Prepared"
    )
    assert controller.run_once()
    assert cluster.get_request("images", second["metadata"]["name"])["status"]["phase"] == (
        "Prepared"
    )
    assert not cluster.jobs
    assert controller.run_once()
    assert cluster.get_request("images", first["metadata"]["name"])["status"]["phase"] == (
        "StartingServer"
    )


def test_completed_untracked_request_is_deleted():
    cluster = FakeCluster()
    cr = make_cr("igr-complete")
    cr["status"] = {"phase": "Succeeded"}
    cluster.requests[("images", cr["metadata"]["name"])] = cr

    assert Controller(cluster, controller_settings()).run_once()

    assert ("images", cr["metadata"]["name"]) in cluster.deleted_requests
    assert ("images", cr["metadata"]["name"]) not in cluster.requests


def test_old_failed_untracked_request_is_deleted_after_retention():
    cluster = FakeCluster()
    cr = make_cr("igr-failed")
    cr["status"] = {"phase": "Failed", "completedAt": "2020-01-01T00:00:00Z"}
    cluster.requests[("images", cr["metadata"]["name"])] = cr

    assert Controller(cluster, controller_settings()).run_once()
    assert ("images", cr["metadata"]["name"]) in cluster.deleted_requests


def test_recent_failed_untracked_request_is_retained_for_error_response():
    cluster = FakeCluster()
    cr = make_cr("igr-failed-recent")
    cr["status"] = {
        "phase": "Failed",
        "completedAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    cluster.requests[("images", cr["metadata"]["name"])] = cr

    assert Controller(cluster, controller_settings()).run_once() is False
    assert ("images", cr["metadata"]["name"]) in cluster.requests


def test_terminal_argocd_managed_requests_are_not_deleted():
    cluster = FakeCluster()
    cr = make_cr("igr-git-managed")
    cr["metadata"]["annotations"] = {
        "argocd.argoproj.io/tracking-id": (
            "llm-gateway:homelab.chik4ge.me/ImageGenerationRequest:"
            "llm-gateway/igr-git-managed"
        )
    }
    cr["status"] = {"phase": "Succeeded"}
    cluster.requests[("images", cr["metadata"]["name"])] = cr

    assert Controller(cluster, controller_settings()).run_once() is False
    assert ("images", cr["metadata"]["name"]) in cluster.requests


def test_openai_request_owning_shared_server_waits_for_other_requests():
    cluster = FakeCluster()
    first = make_cr("igr-openai-complete")
    first["status"] = {"phase": "Succeeded", "serverJobName": "shared-server"}
    second = make_cr("igr-openai-active")
    second["metadata"]["uid"] = "87654321-4321-4321-4321-cba987654321"
    second["status"] = {"phase": "Proxying", "serverJobName": "shared-server"}
    cluster.requests[("images", first["metadata"]["name"])] = first
    cluster.requests[("images", second["metadata"]["name"])] = second
    cluster.jobs[("images", "shared-server")] = {"status": {}}

    Controller(cluster, controller_settings()).run_once()

    assert ("images", first["metadata"]["name"]) in cluster.requests
    assert ("images", first["metadata"]["name"]) not in cluster.deleted_requests
