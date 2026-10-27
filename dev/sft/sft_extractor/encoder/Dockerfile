# Prebuilt image for embedding with Talkie 13B on a cloud GPU (vast.ai).
# Weights are NOT baked in (53GB) — talkie downloads them at runtime to the disk.
# Build from the repo root:
#   docker build -t ghcr.io/<you>/talkie-encoder:latest -f encoder/Dockerfile .
#   echo $GHCR_PAT | docker login ghcr.io -u <you> --password-stdin
#   docker push ghcr.io/<you>/talkie-encoder:latest
# Then make the package Public in GitHub (Packages > settings) so vast can pull it.

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/* && \
    python --version && \
    python -m pip install --no-cache-dir \
    "git+https://github.com/talkie-lm/talkie" numpy

# Put HF/talkie caches on the big instance disk, not the small container layer.
ENV HF_HOME=/workspace/.cache/huggingface \
    TALKIE_CACHE=/workspace/.cache/talkie

WORKDIR /workspace
COPY encoder/ /workspace/encoder/

CMD ["bash"]
