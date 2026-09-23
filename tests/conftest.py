from __future__ import annotations

import copy

from image_gateway.cluster import ConflictError, NotFoundError


class FakeCluster:
    def __init__(self):
        self.requests = {}
        self.jobs = {}
        self.cronjobs = {
            ("images", "image-generation-server"): make_cronjob_template(),
        }
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
        for key, resource in list(self.jobs.items()):
            owners = resource.get("metadata", {}).get("ownerReferences", [])
            if any(owner.get("uid") == uid for owner in owners):
                self.jobs.pop(key)

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
            "suspend": False,
        },
    }


def make_cronjob_template():
    container = {
        "name": "sd-server",
        "image": "stable-diffusion:test",
        "args": ["--listen-port", "8080"],
        "resources": {
            "requests": {"memory": "6Gi", "nvidia.com/gpu": "1"},
            "limits": {"memory": "12Gi", "nvidia.com/gpu": "1"},
        },
        "volumeMounts": [{"name": "models", "mountPath": "/models", "readOnly": True}],
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "CronJob",
        "metadata": {"name": "image-generation-server"},
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
                            "volumes": [
                                {
                                    "name": "models",
                                    "persistentVolumeClaim": {
                                        "claimName": "image-generation-models"
                                    },
                                }
                            ],
                        },
                    },
                },
            },
        },
    }
