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
        self.outputs = {}
        self.scales = []
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
