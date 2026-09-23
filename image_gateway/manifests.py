from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .argocd import tracking_annotations
from .config import Settings
from .naming import (
    diffuser_configmap_name,
    optimizer_configmap_name,
)

GROUP = "homelab.chik4ge.me"
VERSION = "v1alpha1"
KIND = "ImageGenerationRequest"
OPTIMIZER_CRONJOB_NAME = "image-generation-optimizer"
DIFFUSER_CRONJOB_NAME = "image-generation-diffuser"


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


def _input_configmap(
    name: str,
    namespace: str,
    owner: dict[str, Any],
    payload: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "ownerReferences": [owner],
            "labels": {"app.kubernetes.io/part-of": "image-generation"},
            "annotations": tracking_annotations(
                settings.argocd_application_name,
                group="",
                kind="ConfigMap",
                namespace=namespace,
                name=name,
            ),
        },
        "data": {"input.json": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    }


def optimizer_configmap(cr: dict[str, Any], settings: Settings) -> dict[str, Any]:
    spec = cr["spec"]
    return _input_configmap(
        optimizer_configmap_name(cr["metadata"]["uid"]),
        cr["metadata"]["namespace"],
        owner_reference(cr),
        {
            "prompt": spec["prompt"],
            "aspect_ratio": spec.get("aspectRatio", "1:1"),
            "steps": spec.get("steps", 40),
            "seed": spec.get("seed", 42),
        },
        settings,
    )


def diffuser_configmap(
    cr: dict[str, Any], settings: Settings, rewritten_prompt: str, wh_ratio: str
) -> dict[str, Any]:
    spec = cr["spec"]
    key = f"{settings.artifact_prefix.rstrip('/')}/{cr['metadata']['name']}.png"
    return _input_configmap(
        diffuser_configmap_name(cr["metadata"]["uid"]),
        cr["metadata"]["namespace"],
        owner_reference(cr),
        {
            "prompt": rewritten_prompt,
            "wh_ratio": wh_ratio,
            "steps": spec.get("steps", 40),
            "seed": spec.get("seed", 42),
            "artifact_key": key,
        },
        settings,
    )


def job_from_cronjob(
    cr: dict[str, Any],
    settings: Settings,
    cronjob: dict[str, Any],
    *,
    name: str,
    configmap_name: str,
) -> dict[str, Any]:
    cronjob_spec = cronjob.get("spec", {})
    if cronjob_spec.get("suspend") is not True:
        template_name = cronjob.get("metadata", {}).get("name")
        raise ValueError(f"CronJob template {template_name} is not suspended")

    namespace = cr["metadata"]["namespace"]
    job_template = cronjob_spec.get("jobTemplate", {})
    job_spec = deepcopy(job_template.get("spec", {}))
    pod_spec = job_spec.get("template", {}).get("spec", {})
    input_volume = next(
        (
            volume
            for volume in pod_spec.get("volumes", [])
            if volume.get("name") == "input" and "configMap" in volume
        ),
        None,
    )
    if input_volume is None:
        template_name = cronjob.get("metadata", {}).get("name")
        raise ValueError(f"CronJob template {template_name} has no input volume")
    input_volume["configMap"]["name"] = configmap_name

    labels = deepcopy(job_template.get("metadata", {}).get("labels", {}))
    labels.update({"app.kubernetes.io/part-of": "image-generation", "image-job": name})
    template = job_spec.get("template", {})
    pod_metadata = template.setdefault("metadata", {})
    pod_labels = deepcopy(pod_metadata.get("labels", {}))
    pod_labels.update({"job-name": name, "image-job": name})
    pod_metadata["labels"] = pod_labels
    annotations = deepcopy(job_template.get("metadata", {}).get("annotations", {}))
    annotations.update(
        tracking_annotations(
            settings.argocd_application_name,
            group="batch",
            kind="Job",
            namespace=namespace,
            name=name,
        )
    )

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "ownerReferences": [owner_reference(cr)],
            "labels": labels,
            "annotations": annotations,
        },
        "spec": job_spec,
    }
