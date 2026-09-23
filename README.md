# image-generation-gateway

The Python image provides health probes, a FIFO GPU-mode controller, and a thin OpenAI Images API proxy. It authenticates requests, queues a Kubernetes Job, waits for the Job's `sd-server`, and streams the request and response. stable-diffusion.cpp owns OpenAI request validation, image edits, generation, and response formatting.

The gateway forwards the original `/v1/images/generations` JSON body and `/v1/images/edits` multipart body to stable-diffusion.cpp. It does not rewrite or validate prompts, nor parse or reconstruct upstream responses.

When several OpenAI requests are queued, the controller prepares them before starting one `sd-server` Job, then reuses that Job for queued requests. The server's OpenAI handlers serialize generation internally. The Job is deleted after the queue drains, before llama.cpp is restored.

## Run

```sh
python -m pip install ".[dev]"
image-gateway api
image-gateway controller
```

The API listens on port `IMAGE_GATEWAY_PORT` (default `8080`). The controller polls FIFO requests and never processes more than one image-generation request at a time.

## API

- `POST /v1/images/generations` and `POST /v1/images/edits` require `Authorization: Bearer <LLM_GATEWAY_API_KEY>` and are streamed through to the ready `sd-server` Job.
- `/healthz` and `/readyz` are monitoring probes. There is no gateway-specific jobs or artifact HTTP API.

## Controller configuration

`IMAGE_NAMESPACE` selects the namespace for the queue and server Job. Optional settings are `LLM_NAMESPACE`, `LLM_DEPLOYMENT_NAME`, `LLM_POD_LABEL_SELECTOR`, `LLM_STOP_TIMEOUT_SECONDS`, `LLM_RESTORE_TIMEOUT_SECONDS`, and `CONTROLLER_POLL_INTERVAL_SECONDS`.

The API creates temporary `ImageGenerationRequest` queue records and patches their status when proxying finishes. The controller removes completed untracked requests after the shared server is no longer in use, and removes failed requests after a short retention period. Kubernetes garbage collection removes request-owned Jobs. Argo CD manages a suspended stable-diffusion server CronJob as a dormant Job template; the controller creates per-queue Jobs from it without unsuspending the CronJob. The controller uses custom-resource status, the named Deployment scale subresource, the server CronJob, Jobs, and Pods. Deployment manifests and RBAC are maintained in `homelab-applications`.

## Checks

```sh
ruff check .
pytest
```
