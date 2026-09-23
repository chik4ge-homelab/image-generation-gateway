# image-generation-gateway

CPU-only `image-api` and single-consumer `gpu-mode-controller` are shipped in one Python 3.12 image. The mode is selected by the CLI subcommand; llama.cpp, Diffusers, model binaries, and generated image bytes are not included.

The controller launches the prompt optimizer directly in the upstream llama.cpp CUDA image using `llama-completion`; a small init container builds its prompt from `/inputs/input.json`. The diffuser uses a separate image that wraps stable-diffusion.cpp with the `/inputs/input.json` to `/dev/termination-log` contract expected by the controller.

## Run

```sh
python -m pip install ".[dev]"
image-gateway api
image-gateway controller
```

The API listens on port `IMAGE_GATEWAY_PORT` (default `8080`). The controller polls FIFO requests and never processes more than one request at a time.

## API

- `POST /v1/images/jobs` creates an `ImageGenerationRequest` with `spec.suspend=true`.
- `GET /v1/images/jobs/{job_id}` returns the CR state.
- `POST /v1/images/jobs/{job_id}/start` changes `spec.suspend` to `false`.
- `GET /v1/images/jobs/{job_id}/artifact` returns artifact metadata and its object endpoint URL after success.
- `/healthz` and `/readyz` are available for probes.

`job_id` and the CR name are deterministic from the idempotency key. A retry after a create conflict reads the same CR and returns it.

## Controller configuration

The controller requires these environment variables:

`IMAGE_NAMESPACE`, `ARTIFACT_ENDPOINT`, and `ARGOCD_APPLICATION_NAME`.

Optional settings are `ARTIFACT_PREFIX`, `LLM_NAMESPACE`, `LLM_DEPLOYMENT_NAME`, `LLM_POD_LABEL_SELECTOR`, `LLM_STOP_TIMEOUT_SECONDS`, `LLM_RESTORE_TIMEOUT_SECONDS`, and `CONTROLLER_POLL_INTERVAL_SECONDS`.

The optimizer writes its completion to `/dev/termination-log`; the controller extracts and validates the `rewritten_prompt` and `wh_ratio` JSON object. The diffuser receives input through `/inputs/input.json` and writes exactly `key`, `sha256`, `content_type`, `width`, and `height` to `/dev/termination-log`, with a maximum serialized size of 4 KiB. The diffuser uploads the image directly to the object bucket; the controller verifies the returned key and SHA-256 through the OBC-provided S3 credentials when they are present, falling back to an unauthenticated HTTP endpoint otherwise.

The API process only uses custom-resource create/get/patch operations. Argo CD manages suspended optimizer and diffuser CronJobs as dormant Job templates; the controller reads only their templates and creates Argo-tracked per-request Jobs without unsuspending the CronJobs. The controller uses custom-resource get/list/watch-equivalent polling and status updates, the named Deployment scale subresource, CronJobs, Jobs, Pods, and ConfigMaps. Deployment manifests and RBAC are intentionally left to `homelab-applications`.

## Checks

```sh
ruff check .
pytest
```
