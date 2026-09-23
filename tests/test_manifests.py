from image_gateway.config import Settings
from image_gateway.manifests import job_from_cronjob, optimizer_configmap
from image_gateway.naming import diffuser_job_name, optimizer_job_name, server_job_name

from .conftest import make_cr, make_cronjob_template


def test_jobs_clone_argocd_managed_suspended_cronjob_templates():
    settings = Settings(
        artifact_endpoint="https://objects.example.test",
    )
    cr = make_cr()
    optimizer_template = make_cronjob_template("optimizer")
    diffuser_template = make_cronjob_template("diffuser")
    optimizer = job_from_cronjob(
        cr,
        settings,
        optimizer_template,
        name=optimizer_job_name(cr["metadata"]["uid"]),
        configmap_name="optimizer-input",
    )
    diffuser = job_from_cronjob(
        cr,
        settings,
        diffuser_template,
        name=diffuser_job_name(cr["metadata"]["uid"]),
        configmap_name="diffuser-input",
    )

    assert optimizer["metadata"]["name"] == optimizer_job_name(cr["metadata"]["uid"])
    assert diffuser["metadata"]["name"] == diffuser_job_name(cr["metadata"]["uid"])
    assert "annotations" not in optimizer["metadata"]
    configmap = optimizer_configmap(cr, settings)
    assert "annotations" not in configmap["metadata"]
    for job in (optimizer, diffuser):
        spec = job["spec"]
        assert spec["backoffLimit"] == 0
        assert spec["activeDeadlineSeconds"] > 0
        assert spec["ttlSecondsAfterFinished"] > 0
        assert spec["template"]["spec"]["restartPolicy"] == "Never"
        assert "command" not in spec["template"]["spec"]["containers"][0]
        assert job["metadata"]["ownerReferences"][0]["name"] == cr["metadata"]["name"]
        assert job["metadata"]["labels"]["image-job"] == job["metadata"]["name"]
        input_volume = next(v for v in spec["template"]["spec"]["volumes"] if v["name"] == "input")
        assert input_volume["configMap"]["name"] in {"optimizer-input", "diffuser-input"}
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
    assert optimizer_resources["limits"]["memory"] == "4Gi"
    assert diffuser_resources["requests"]["memory"] == "6Gi"
    assert diffuser_resources["limits"]["memory"] == "12Gi"
    assert optimizer_template["spec"]["suspend"] is True
    optimizer_input = optimizer_template["spec"]["jobTemplate"]["spec"]["template"]["spec"][
        "volumes"
    ][0]["configMap"]["name"]
    assert optimizer_input == "image-generation-optimizer-input-template"


def test_running_cronjob_cannot_be_used_as_a_template():
    settings = Settings(artifact_endpoint="https://objects.example.test")
    cronjob = make_cronjob_template("optimizer")
    cronjob["spec"]["suspend"] = False

    try:
        job_from_cronjob(
            make_cr(), settings, cronjob, name="igr-opt-test", configmap_name="optimizer-input"
        )
    except ValueError as exc:
        assert "not suspended" in str(exc)
    else:
        raise AssertionError("an active CronJob must not be used as a template")


def test_server_job_has_no_request_configmap_or_argocd_tracking():
    settings = Settings(artifact_endpoint="https://objects.example.test")
    cr = make_cr()
    cr["metadata"]["namespace"] = "images"
    server = job_from_cronjob(
        cr,
        settings,
        make_cronjob_template("server"),
        name=server_job_name(cr["metadata"]["uid"]),
        configmap_name=None,
    )

    pod_spec = server["spec"]["template"]["spec"]
    assert server["metadata"]["name"] == server_job_name(cr["metadata"]["uid"])
    assert all(volume["name"] != "input" for volume in pod_spec["volumes"])
    assert "annotations" not in server["metadata"]
