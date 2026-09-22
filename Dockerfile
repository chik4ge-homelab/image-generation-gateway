FROM python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9 AS gateway

WORKDIR /app
COPY pyproject.toml README.md ./
COPY image_gateway ./image_gateway

RUN pip install --no-cache-dir .

RUN addgroup --system app && adduser --system --ingroup app app
USER app

ENTRYPOINT ["image-gateway"]
CMD ["api"]

FROM ghcr.io/ggml-org/llama.cpp:full-cuda@sha256:1276d8b3f09fe1cb05b8d7369f64a5a9c15619fb05a61f3df1d261a4b53c96d9 AS optimizer

COPY workers/optimizer.py /usr/local/bin/image-optimizer
ENTRYPOINT ["python3", "/usr/local/bin/image-optimizer"]

FROM ghcr.io/leejet/stable-diffusion.cpp:master-cuda@sha256:b0b24683e7020cdb2f5c0e36805abde9b9c6ae1eda6f9d305c73b07eabc3669a AS diffuser

RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip \
    && python3 -m pip install --break-system-packages --no-cache-dir boto3 \
    && rm -rf /var/lib/apt/lists/*

COPY workers/diffuser.py /usr/local/bin/image-diffuser
ENTRYPOINT ["python3", "/usr/local/bin/image-diffuser"]
