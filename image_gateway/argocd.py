from __future__ import annotations


def tracking_annotations(
    application: str,
    *,
    group: str,
    kind: str,
    namespace: str,
    name: str,
) -> dict[str, str]:
    if not application:
        return {}
    return {
        "argocd.argoproj.io/tracking-id": (
            f"{application}:{group}/{kind}:{namespace}/{name}"
        ),
        "argocd.argoproj.io/compare-options": "IgnoreExtraneous",
        "argocd.argoproj.io/sync-options": "Prune=false",
    }
