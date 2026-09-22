from __future__ import annotations

import base64
import hashlib
import re


def request_name(idempotency_key: str) -> str:
    digest = hashlib.sha256(idempotency_key.encode("utf-8")).digest()
    encoded = base64.b32encode(digest).decode("ascii").rstrip("=").lower()
    return f"igr-{encoded}"


def uid_name(prefix: str, uid: str) -> str:
    safe_uid = re.sub(r"[^a-z0-9-]", "-", uid.lower()).strip("-")
    return f"{prefix}-{safe_uid}"[:63].rstrip("-")


def optimizer_job_name(uid: str) -> str:
    return uid_name("igr-opt", uid)


def diffuser_job_name(uid: str) -> str:
    return uid_name("igr-diff", uid)


def optimizer_configmap_name(uid: str) -> str:
    return uid_name("igr-opt-input", uid)


def diffuser_configmap_name(uid: str) -> str:
    return uid_name("igr-diff-input", uid)
