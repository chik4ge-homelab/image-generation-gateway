from __future__ import annotations

import copy
import hashlib
import json

from image_gateway.cluster import ConflictError, NotFoundError


class FakeCluster:
    def __init__(self):
        self.requests = {}
        self.configmaps = {}
        self.jobs = {}
        self.cronjobs = {
            ("images", "image-generation-optimizer"): make_cronjob_template("optimizer"),
            ("images", "image-generation-diffuser"): make_cronjob_template("diffuser"),
            ("images", "image-generation-server"): make_cronjob_template("server"),
        }
        self.outputs = {}
        self.scales = []
        self.deleted_jobs = set()
        self.deleted_requests = set()
        self.job_endpoints = {}
        self.pods_gone = True
        self.uid = "12345678-1234-1234-1234-123456789abc"
        self._rv = 1

    def ready(self):
        return True

    def get_request(self, namespace, name):
        try:
            return copy.deepcopy(self.requests[(namespace, name)])
        except KeyError as exc:
            raise NotFoundError(name) from exc

    def create_request(self, namespace, body):
        key = (namespace, body["metadata"]["name"])
        if key in self.requests:
            raise ConflictError(body["metadata"]["name"])
        result = copy.deepcopy(body)
        result["metadata"].update(
            {
                "uid": self.uid,
                "resourceVersion": str(self._rv),
                "creationTimestamp": "2026-01-01T00:00:00Z",
            }
        )
        self._rv += 1
        self.requests[key] = result
        return copy.deepcopy(result)

    def patch_request(self, namespace, name, patch):
        result = self.get_request(namespace, name)
        for key, value in patch.items():
            if isinstance(value, dict):
                result.setdefault(key, {}).update(value)
            else:
                result[key] = value
        result["metadata"]["resourceVersion"] = str(self._rv)
        self._rv += 1
        self.requests[(namespace, name)] = result
        return copy.deepcopy(result)

    def list_requests(self, namespace):
        return [copy.deepcopy(value) for (ns, _), value in self.requests.items() if ns == namespace]

    def delete_request(self, namespace, name):
        request = self.requests.pop((namespace, name), None)
        if request is None:
            return
        self.deleted_requests.add((namespace, name))
        uid = request["metadata"].get("uid")
        for resources in (self.configmaps, self.jobs):
            for key, resource in list(resources.items()):
                owners = resource.get("metadata", {}).get("ownerReferences", [])
                if any(owner.get("uid") == uid for owner in owners):
                    resources.pop(key)

    def patch_request_status(self, namespace, name, status, resource_version=None):
        result = self.get_request(namespace, name)
        if (
            resource_version is not None
            and result["metadata"]["resourceVersion"] != resource_version
        ):
            raise ConflictError("resource version conflict")
        result.setdefault("status", {}).update(copy.deepcopy(status))
        result["metadata"]["resourceVersion"] = str(self._rv)
        self._rv += 1
        self.requests[(namespace, name)] = result
        return copy.deepcopy(result)

    def get_configmap(self, namespace, name):
        try:
            return copy.deepcopy(self.configmaps[(namespace, name)])
        except KeyError as exc:
            raise NotFoundError(name) from exc

    def create_configmap(self, namespace, body):
        key = (namespace, body["metadata"]["name"])
        if key in self.configmaps:
            raise ConflictError(body["metadata"]["name"])
        self.configmaps[key] = copy.deepcopy(body)
        return copy.deepcopy(body)

    def get_job(self, namespace, name):
        try:
            return copy.deepcopy(self.jobs[(namespace, name)])
        except KeyError as exc:
            raise NotFoundError(name) from exc

    def create_job(self, namespace, body):
        key = (namespace, body["metadata"]["name"])
        if key in self.jobs:
            raise ConflictError(body["metadata"]["name"])
        result = copy.deepcopy(body)
        result.setdefault("status", {})
        self.jobs[key] = result
        return copy.deepcopy(result)

    def delete_job(self, namespace, name):
        self.deleted_jobs.add((namespace, name))

    def job_pods_gone(self, namespace, job_name):
        return (namespace, job_name) in self.deleted_jobs

    def wait_job_pods_gone(self, namespace, job_name, timeout_seconds):
        return self.job_pods_gone(namespace, job_name)

    def job_endpoint(self, namespace, job_name, port):
        return self.job_endpoints.get((namespace, job_name))

    def get_cronjob(self, namespace, name):
        try:
            return copy.deepcopy(self.cronjobs[(namespace, name)])
        except KeyError as exc:
            raise NotFoundError(name) from exc

    def llm_scale(self, namespace, name, replicas):
        self.scales.append((namespace, name, replicas))

    def llm_pods_gone(self, namespace, name):
        return self.pods_gone

    def wait_llm_ready(self, namespace, name, timeout_seconds):
        return True

    def read_job_output(self, namespace, job_name):
        return self.outputs.get((namespace, job_name))

    def verify_artifact(self, endpoint, artifact):
        return True

    def artifact_url(self, endpoint, artifact):
        return endpoint.rstrip("/") + "/" + artifact["key"]


def make_cr(name="igr-test"):
    return {
        "apiVersion": "homelab.chik4ge.me/v1alpha1",
        "kind": "ImageGenerationRequest",
        "metadata": {
            "name": name,
            "namespace": "images",
            "uid": "12345678-1234-1234-1234-123456789abc",
            "resourceVersion": "1",
            "creationTimestamp": "2026-01-01T00:00:00Z",
        },
        "spec": {
            "idempotencyKey": "key",
            "prompt": "a cat",
            "aspectRatio": "1:1",
            "steps": 40,
            "seed": 42,
            "suspend": False,
        },
    }


def make_cronjob_template(worker):
    is_optimizer = worker == "optimizer"
    is_server = worker == "server"
    volumes = [
        {
            "name": "models",
            "persistentVolumeClaim": {"claimName": "image-generation-models"},
        }
    ]
    if not is_server:
        volumes.insert(
            0,
            {
                "name": "input",
                "configMap": {
                    "name": f"image-generation-{worker}-input-template",
                    "items": [{"key": "input.json", "path": "input.json"}],
                },
            },
        )
    container = {
        "name": worker,
        "image": f"{worker}:test",
        "args": ["--input", "/inputs/input.json"] if not is_server else ["--listen-port", "8080"],
        "env": [{"name": "MODEL_PATH", "value": "/models/model.gguf"}],
        "resources": {
            "requests": {
                "memory": "4Gi" if is_optimizer else "6Gi",
                "nvidia.com/gpu": "1",
            },
            "limits": {
                "memory": "4Gi" if is_optimizer else "12Gi",
                "nvidia.com/gpu": "1",
            },
        },
        "volumeMounts": [
            *(
                [{"name": "input", "mountPath": "/inputs", "readOnly": True}]
                if not is_server
                else []
            ),
            {"name": "models", "mountPath": "/models", "readOnly": True},
        ],
    }
    if not is_optimizer:
        container["envFrom"] = [{"secretRef": {"name": "artifacts"}}]
    return {
        "apiVersion": "batch/v1",
        "kind": "CronJob",
        "metadata": {"name": f"image-generation-{worker}"},
        "spec": {
            "suspend": True,
            "jobTemplate": {
                "metadata": {"labels": {"app.kubernetes.io/part-of": "image-generation"}},
                "spec": {
                    "backoffLimit": 0,
                    "activeDeadlineSeconds": 3600,
                    "ttlSecondsAfterFinished": 3600,
                    "template": {
                        "metadata": {"labels": {"app.kubernetes.io/part-of": "image-generation"}},
                        "spec": {
                            "restartPolicy": "Never",
                            "serviceAccountName": "image-generation-job",
                            "runtimeClassName": "nvidia",
                                    "containers": [container],
                                    "volumes": volumes,
                        },
                    },
                },
            },
        },
    }


def artifact_output():
    data = b"image-bytes"
    return json.dumps(
        {
            "key": "image-generation/igr-test.png",
            "sha256": hashlib.sha256(data).hexdigest(),
            "content_type": "image/png",
            "width": 512,
            "height": 512,
        }
    )
