from image_gateway.manifests import job_from_cronjob
from image_gateway.naming import server_job_name

from .conftest import make_cr, make_cronjob_template


def test_server_job_clones_the_suspended_cronjob_template():
    cr = make_cr()
    cronjob = make_cronjob_template()
    server = job_from_cronjob(cr, cronjob, name=server_job_name(cr["metadata"]["uid"]))

    spec = server["spec"]
    assert server["metadata"]["name"] == server_job_name(cr["metadata"]["uid"])
    assert "annotations" not in server["metadata"]
    assert spec["backoffLimit"] == 0
    assert spec["activeDeadlineSeconds"] > 0
    assert spec["ttlSecondsAfterFinished"] > 0
    assert spec["template"]["spec"]["restartPolicy"] == "Never"
    assert spec["template"]["spec"]["runtimeClassName"] == "nvidia"
    assert spec["template"]["spec"]["containers"][0]["resources"]["limits"][
        "nvidia.com/gpu"
    ] == "1"
    assert server["metadata"]["ownerReferences"][0]["name"] == cr["metadata"]["name"]
    assert server["metadata"]["labels"]["image-job"] == server["metadata"]["name"]
    assert all(volume["name"] != "input" for volume in spec["template"]["spec"]["volumes"])
    assert cronjob["spec"]["suspend"] is True


def test_running_cronjob_cannot_be_used_as_a_template():
    cronjob = make_cronjob_template()
    cronjob["spec"]["suspend"] = False

    try:
        job_from_cronjob(make_cr(), cronjob, name="igr-server-test")
    except ValueError as exc:
        assert "not suspended" in str(exc)
    else:
        raise AssertionError("an active CronJob must not be used as a template")
