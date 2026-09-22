from __future__ import annotations

import json
from typing import Any

from .config import Settings
from .naming import (
    diffuser_configmap_name,
    diffuser_job_name,
    optimizer_configmap_name,
    optimizer_job_name,
)

GROUP = "homelab.chik4ge.me"
VERSION = "v1alpha1"
KIND = "ImageGenerationRequest"


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
    name: str, namespace: str, owner: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "ownerReferences": [owner],
            "labels": {"app.kubernetes.io/part-of": "image-generation"},
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
    )


def _common_job(
    *,
    cr: dict[str, Any],
    settings: Settings,
    name: str,
    configmap_name: str,
    container_name: str,
    image: str,
    model_path: str,
    args: list[str],
    gpu: bool,
    artifact_credentials: bool,
) -> dict[str, Any]:
    namespace = cr["metadata"]["namespace"]
    env = [
        {"name": "INPUT_PATH", "value": "/inputs/input.json"},
        {"name": "OUTPUT_PATH", "value": "/dev/termination-log"},
        {"name": "MODEL_PATH", "value": model_path},
        {"name": "ARTIFACT_ENDPOINT", "value": settings.artifact_endpoint},
    ]
    container: dict[str, Any] = {
        "name": container_name,
        "image": image,
        "imagePullPolicy": "IfNotPresent",
        "args": args,
        "env": env,
        "volumeMounts": [
            {"name": "input", "mountPath": "/inputs", "readOnly": True},
            {"name": "models", "mountPath": "/models", "readOnly": True},
        ],
    }
    if artifact_credentials:
        container["envFrom"] = [{"secretRef": {"name": settings.object_bucket_secret_name}}]
    if gpu:
        gpu_resources = {settings.gpu_resource_name: str(settings.gpu_count)}
        if container_name == "diffuser":
            container["resources"] = {
                "requests": {"cpu": "2", "memory": "6Gi", **gpu_resources},
                "limits": {"memory": "12Gi", **gpu_resources},
            }
        else:
            container["resources"] = {
                "requests": {"cpu": "2", "memory": "4Gi", **gpu_resources},
                "limits": {"memory": "10Gi", **gpu_resources},
            }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "ownerReferences": [owner_reference(cr)],
            "labels": {"app.kubernetes.io/part-of": "image-generation", "image-job": name},
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": settings.job_active_deadline_seconds,
            "ttlSecondsAfterFinished": settings.job_ttl_seconds,
            "template": {
                "metadata": {"labels": {"job-name": name, "image-job": name}},
                "spec": {
                    "restartPolicy": "Never",
                    "serviceAccountName": settings.job_service_account_name,
                    "runtimeClassName": settings.gpu_runtime_class_name,
                    "containers": [container],
                    "volumes": [
                        {
                            "name": "input",
                            "configMap": {
                                "name": configmap_name,
                                "items": [{"key": "input.json", "path": "input.json"}],
                            },
                        },
                        {
                            "name": "models",
                            "persistentVolumeClaim": {"claimName": settings.model_pvc_name},
                        },
                    ],
                },
            },
        },
    }


def optimizer_job(cr: dict[str, Any], settings: Settings) -> dict[str, Any]:
    uid = cr["metadata"]["uid"]
    return _common_job(
        cr=cr,
        settings=settings,
        name=optimizer_job_name(uid),
        configmap_name=optimizer_configmap_name(uid),
        container_name="optimizer",
        image=settings.optimizer_image,
        model_path=settings.optimizer_model_path,
        args=["--input", "/inputs/input.json", "--output", "/dev/termination-log"],
        gpu=True,
        artifact_credentials=False,
    )


def diffuser_job(cr: dict[str, Any], settings: Settings) -> dict[str, Any]:
    uid = cr["metadata"]["uid"]
    return _common_job(
        cr=cr,
        settings=settings,
        name=diffuser_job_name(uid),
        configmap_name=diffuser_configmap_name(uid),
        container_name="diffuser",
        image=settings.diffuser_image,
        model_path=settings.diffuser_model_path,
        args=["--input", "/inputs/input.json", "--output", "/dev/termination-log"],
        gpu=True,
        artifact_credentials=True,
    )
