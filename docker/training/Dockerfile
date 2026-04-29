FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ARG LLM_STUDIO_IMAGE_REVISION=unknown
ARG LLM_STUDIO_IMAGE_BUILT_AT=unknown

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    LLM_STUDIO_REMOTE_WORKSPACE=/workspace/llm-studio \
    LLM_STUDIO_IMAGE_REVISION=${LLM_STUDIO_IMAGE_REVISION} \
    LLM_STUDIO_IMAGE_BUILT_AT=${LLM_STUDIO_IMAGE_BUILT_AT}

WORKDIR /opt/llm-builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential zstd ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY apps/llm-studio/api/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements.txt zstandard

COPY model ./model
COPY tokenizer ./tokenizer
COPY training ./training
COPY llm_builder ./llm_builder
COPY apps/llm-studio/remote_agent ./remote_agent

RUN python -c "import shutil; assert shutil.which('gcc'); import llm_builder.local_text_data; import training.runner"

COPY docker/training/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8021
ENTRYPOINT ["/entrypoint.sh"]
