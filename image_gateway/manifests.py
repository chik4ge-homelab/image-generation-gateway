from __future__ import annotations

from copy import deepcopy
from typing import Any

GROUP = "homelab.chik4ge.me"
VERSION = "v1alpha1"
KIND = "ImageGenerationRequest"
SERVER_CRONJOB_NAME = "image-generation-server"


def owner_reference(cr: dict[str, Any]) -> dict[str, Any]:
    metadata = cr["metadata"]
    return {
        "apiVersion": f"{GROUP}/{VERSION}",
        "kind": KIND,
        "name": metadata["name"],
        "uid": metadata["uid"],
        "controller": True,
        "blockOwnerDeletion": True,
    }


def job_from_cronjob(
    cr: dict[str, Any],
    cronjob: dict[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    cronjob_spec = cronjob.get("spec", {})
    if cronjob_spec.get("suspend") is not True:
        template_name = cronjob.get("metadata", {}).get("name")
        raise ValueError(f"CronJob template {template_name} is not suspended")

    namespace = cr["metadata"]["namespace"]
    job_template = cronjob_spec.get("jobTemplate", {})
    job_spec = deepcopy(job_template.get("spec", {}))

    labels = deepcopy(job_template.get("metadata", {}).get("labels", {}))
    labels.update({"app.kubernetes.io/part-of": "image-generation", "image-job": name})
    template = job_spec.get("template", {})
    pod_metadata = template.setdefault("metadata", {})
    pod_labels = deepcopy(pod_metadata.get("labels", {}))
    pod_labels.update({"job-name": name, "image-job": name})
    pod_metadata["labels"] = pod_labels
    annotations = deepcopy(job_template.get("metadata", {}).get("annotations", {}))

    metadata = {
        "name": name,
        "namespace": namespace,
        "ownerReferences": [owner_reference(cr)],
        "labels": labels,
    }
    if annotations:
        metadata["annotations"] = annotations

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": metadata,
        "spec": job_spec,
    }
