# Image for rewind.platform.sandbox.DockerSandbox: a plain Python runtime,
# nothing else. The sandbox's isolation comes from the container flags
# (SandboxPolicy.container_flags), not from this image.
#
#   docker build -t rewind-sandbox:py -f docker/sandbox.Dockerfile .
FROM python:3.12-slim
RUN ln -sf /usr/local/bin/python3 /usr/bin/python3 \
 && rm -rf /usr/local/lib/python3.12/ensurepip /usr/local/lib/python3.12/idlelib /root/.cache
USER 65534:65534
WORKDIR /work
