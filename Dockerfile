FROM python:3.12-slim

# Installer curl
RUN apt-get update && apt-get install -y curl && apt-get clean
RUN apt-get update && apt-get install -y git && apt-get clean
RUN apt-get update && apt-get install -y iputils-ping
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    gcc \
    cmake \
    make \
    && rm -rf /var/lib/apt/lists/*

# install ffmpeg
RUN apt update && \
    apt install -y ffmpeg 

# Set environment variables for Hugging Face cache location
ENV HF_HOME=/app/model_weights
ENV HF_HUB_CACHE=/app/model_weights/hub


# Set workdir inside the container
WORKDIR /app/ragondin

# Install uv & setup venv
COPY pyproject.toml uv.lock ./
RUN pip3 install uv && \
    uv sync

# Copy source code
COPY ragondin/ .

# Copy assests & config
COPY public/ /app/public/
COPY prompts/ /app/prompts/
COPY .hydra_config/ /app/.hydra_config/
ENV PYTHONPATH=/app/ragondin/