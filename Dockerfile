# This is an example Dockerfile that builds a minimal container for running LK Agents
# For more information on the build process, see https://docs.livekit.io/agents/ops/deployment/builds/
# syntax=docker/dockerfile:1

# Use the official UV Python base image with Python 3.13 on Debian Bookworm
# UV is a fast Python package manager that provides better performance than pip
# We use the slim variant to keep the image size smaller while still having essential tools
ARG PYTHON_VERSION=3.13
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS base

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

# Install build dependencies required for Python packages with native extensions
# gcc: C compiler needed for building Python packages with C extensions
# g++: C++ compiler needed for building Python packages with C++ extensions
# python3-dev: Python development headers needed for compilation
# We clean up the apt cache after installation to keep the image size down
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*

# Create a new directory for our application code
# And set it as the working directory
WORKDIR /app

# Copy all files first (needed for workspace packages like sub200)
# This includes source code, workspace directories, and configuration files
COPY . .

# Remove workspace directories we want to install from PyPI instead
# Keep only livekit-plugins-sub200 since it's a local plugin
RUN rm -rf livekit-agents livekit-plugins/livekit-plugins-silero livekit-plugins/livekit-plugins-turn-detector livekit-plugins/livekit-plugins-openai

# Remove workspace source overrides for packages available on PyPI
# Keep sub200 as workspace source since it's a local plugin
RUN sed -i '/^livekit-agents = { workspace = true }$/d; /^livekit-plugins-silero = { workspace = true }$/d; /^livekit-plugins-turn-detector = { workspace = true }$/d; /^livekit-plugins-openai = { workspace = true }$/d' pyproject.toml

# Regenerate lock file and install dependencies
# sub200 will be built from workspace, others from PyPI
RUN uv lock && uv sync --locked

# Change ownership of all app files to the non-privileged user
# This ensures the application can read/write files as needed
RUN chown -R appuser:appuser /app

# Switch to the non-privileged user for all subsequent operations
# This improves security by not running as root
USER appuser

# Pre-download any ML models or files the agent needs
# This ensures the container is ready to run immediately without downloading
# dependencies at runtime, which improves startup time and reliability
RUN uv run "tamil_agent.py" download-files

# Run the application using UV
# UV will activate the virtual environment and run the agent.
# The "start" command tells the worker to connect to LiveKit and begin waiting for jobs.
CMD ["uv", "run", "tamil_agent.py", "start"]
