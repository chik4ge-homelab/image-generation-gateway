# image-generation-gateway

The Python image provides health probes, a FIFO GPU-mode controller, and a thin OpenAI Images API proxy. It authenticates requests, queues a Kubernetes Job, waits for the Job's `sd-server`, and streams the request and response. stable-diffusion.cpp owns OpenAI request validation, image edits, generation, and response formatting.

For valid string prompts on `/v1/images/generations`, the controller runs the existing prompt optimizer first; the proxy changes only the `prompt` value before forwarding. `/v1/images/edits` bodies are forwarded unchanged so multipart reference images and masks remain upstream concerns. Other OpenAI fields and upstream responses are not parsed or reconstructed by this gateway.

When several OpenAI requests are already queued, the controller prepares their prompts before starting one `sd-server` Job, then reuses that Job for the queued requests. The server's OpenAI handlers serialize generation internally. It is deleted after the queue drains, before llama.cpp is restored.

## Run

```sh
python -m pip install ".[dev]"
image-gateway api
image-gateway controller
```

The API listens on port `IMAGE_GATEWAY_PORT` (default `8080`). The controller polls FIFO requests and never processes more than one request at a time.

## API

- `POST /v1/images/generations` and `POST /v1/images/edits` require `Authorization: Bearer <LLM_GATEWAY_API_KEY>` and are streamed through to the ready `sd-server` Job.
- `/healthz` and `/readyz` are monitoring probes. There is no gateway-specific jobs or artifact HTTP API.

## Controller configuration

The controller requires these environment variables:

`IMAGE_NAMESPACE`, `ARTIFACT_ENDPOINT`, and `ARGOCD_APPLICATION_NAME`.

Optional settings are `ARTIFACT_PREFIX`, `LLM_NAMESPACE`, `LLM_DEPLOYMENT_NAME`, `LLM_POD_LABEL_SELECTOR`, `LLM_STOP_TIMEOUT_SECONDS`, `LLM_RESTORE_TIMEOUT_SECONDS`, and `CONTROLLER_POLL_INTERVAL_SECONDS`.

The optimizer writes its completion to `/dev/termination-log`; the controller extracts and validates the `rewritten_prompt` and `wh_ratio` JSON object. The diffuser receives input through `/inputs/input.json` and writes exactly `key`, `sha256`, `content_type`, `width`, and `height` to `/dev/termination-log`, with a maximum serialized size of 4 KiB. The diffuser uploads the image directly to the object bucket; the controller verifies the returned key and SHA-256 through the OBC-provided S3 credentials when they are present, falling back to an unauthenticated HTTP endpoint otherwise.

The API process creates and reads `ImageGenerationRequest` records and patches their status when proxying finishes. Argo CD manages suspended optimizer, diffuser, and stable-diffusion server CronJobs as dormant Job templates; the controller creates Argo-tracked per-request Jobs without unsuspending the CronJobs. The controller uses custom-resource polling/status updates, the named Deployment scale subresource, CronJobs, Jobs, Pods, and ConfigMaps. Deployment manifests and RBAC are intentionally left to `homelab-applications`.

## Checks

```sh
ruff check .
pytest
```
