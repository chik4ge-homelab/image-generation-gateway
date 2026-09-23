from image_gateway.config import Settings
from image_gateway.manifests import diffuser_job, optimizer_configmap, optimizer_job
from image_gateway.naming import diffuser_job_name, optimizer_job_name

from .conftest import make_cr


def test_job_manifests_are_deterministic_and_have_safety_limits():
    settings = Settings(
        optimizer_image="optimizer:test",
        diffuser_image="diffuser:test",
        optimizer_model_path="/models/optimizer.gguf",
        diffuser_model_path="/models/diffuser.safetensors",
        object_bucket_secret_name="object-bucket",
        artifact_endpoint="https://objects.example.test",
        argocd_application_name="llm-gateway",
    )
    cr = make_cr()
    optimizer = optimizer_job(cr, settings)
    diffuser = diffuser_job(cr, settings)

    assert optimizer["metadata"]["name"] == optimizer_job_name(cr["metadata"]["uid"])
    assert diffuser["metadata"]["name"] == diffuser_job_name(cr["metadata"]["uid"])
    assert optimizer["metadata"]["annotations"] == {
        "argocd.argoproj.io/tracking-id": (
            f"llm-gateway:batch/Job:images/{optimizer['metadata']['name']}"
        ),
        "argocd.argoproj.io/compare-options": "IgnoreExtraneous",
        "argocd.argoproj.io/sync-options": "Prune=false",
    }
    configmap = optimizer_configmap(cr, settings)
    assert configmap["metadata"]["annotations"]["argocd.argoproj.io/tracking-id"].startswith(
        "llm-gateway:/ConfigMap:images/"
    )
    for job in (optimizer, diffuser):
        spec = job["spec"]
        assert spec["backoffLimit"] == 0
        assert spec["activeDeadlineSeconds"] > 0
        assert spec["ttlSecondsAfterFinished"] > 0
        assert spec["template"]["spec"]["restartPolicy"] == "Never"
        assert "command" not in spec["template"]["spec"]["containers"][0]
    for job in (optimizer, diffuser):
        assert job["spec"]["template"]["spec"]["runtimeClassName"] == "nvidia"
        container = job["spec"]["template"]["spec"]["containers"][0]
        assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
        assert any(
            mount["mountPath"] == "/models" and mount["readOnly"] is True
            for mount in container["volumeMounts"]
        )
    assert "envFrom" not in optimizer["spec"]["template"]["spec"]["containers"][0]
    assert diffuser["spec"]["template"]["spec"]["containers"][0]["envFrom"]
    optimizer_resources = optimizer["spec"]["template"]["spec"]["containers"][0]["resources"]
    diffuser_resources = diffuser["spec"]["template"]["spec"]["containers"][0]["resources"]
    assert optimizer_resources["requests"]["memory"] == "4Gi"
    assert optimizer_resources["limits"]["memory"] == "12Gi"
    assert diffuser_resources["requests"]["memory"] == "6Gi"
    assert diffuser_resources["limits"]["memory"] == "12Gi"
    optimizer_env = optimizer["spec"]["template"]["spec"]["containers"][0]["env"]
    diffuser_env = diffuser["spec"]["template"]["spec"]["containers"][0]["env"]
    assert {item["name"]: item["value"] for item in optimizer_env}["MODEL_PATH"] == (
        "/models/optimizer.gguf"
    )
    assert {item["name"]: item["value"] for item in diffuser_env}["MODEL_PATH"] == (
        "/models/diffuser.safetensors"
    )
