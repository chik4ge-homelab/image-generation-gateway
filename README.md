# image-generation-gateway

The Python image provides health probes, a FIFO GPU-mode controller, and a thin OpenAI Images API proxy. It authenticates requests, queues a Kubernetes Job, waits for the Job's `sd-server`, and streams the request and response. stable-diffusion.cpp owns OpenAI request validation, image edits, generation, and response formatting.

The gateway does not rewrite or validate generation prompts. It forwards the original `/v1/images/generations` JSON body and `/v1/images/edits` multipart body to stable-diffusion.cpp, leaving request validation, prompt handling, image generation, and response formatting upstream. Other OpenAI fields and upstream responses are not parsed or reconstructed by this gateway.

When several OpenAI requests are already queued, the controller prepares them before starting one `sd-server` Job, then reuses that Job for the queued requests. The server's OpenAI handlers serialize generation internally. It is deleted after the queue drains, before llama.cpp is restored.

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

`IMAGE_NAMESPACE` and `ARTIFACT_ENDPOINT`.

Optional settings are `ARTIFACT_PREFIX`, `LLM_NAMESPACE`, `LLM_DEPLOYMENT_NAME`, `LLM_POD_LABEL_SELECTOR`, `LLM_STOP_TIMEOUT_SECONDS`, `LLM_RESTORE_TIMEOUT_SECONDS`, and `CONTROLLER_POLL_INTERVAL_SECONDS`.

The older artifact-mode CR path still uses optimizer and diffuser CronJobs and object storage. The public OpenAI image routes create `openai` requests and bypass that pipeline, forwarding the original request directly to stable-diffusion.cpp.

The API creates temporary `ImageGenerationRequest` queue records and patches their status when proxying finishes. The controller removes completed untracked requests after the shared server is no longer in use, and removes failed requests after a short retention period. Kubernetes garbage collection then removes request-owned Jobs and ConfigMaps. Argo CD manages suspended optimizer, diffuser, and stable-diffusion server CronJobs as dormant Job templates; the controller creates untracked per-request Jobs from those templates without unsuspending the CronJobs. The controller uses custom-resource polling/status updates, the named Deployment scale subresource, CronJobs, Jobs, Pods, and ConfigMaps. Deployment manifests and RBAC are intentionally left to `homelab-applications`.

## Checks

```sh
ruff check .
pytest
```
