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
ENV XDG_CACHE_HOME=${XDG_CACHE_HOME:-/app/model_weights}
ENV HF_HOME=${HF_HOME:-/app/model_weights}
ENV HF_HUB_CACHE=${HF_HUB_CACHE:-/app/model_weights/hub}

# Set workdir for uv
WORKDIR /app

# Install uv & setup venv
COPY pyproject.toml uv.lock ./
RUN pip3 install uv && \
    uv python install 3.12.7 && \
    uv python pin 3.12.7 && \
    uv sync --no-dev
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
# Set workdir for source code
WORKDIR /app/ragondin

# Copy source code
COPY ragondin/ .

# Copy assests & config
COPY public/ /app/public/
COPY prompts/ /app/prompts/
COPY .hydra_config/ /app/.hydra_config/
ENV PYTHONPATH=/app/ragondin/
ENV APP_iPORT=${APP_iPORT:-8080}
ENTRYPOINT ../entrypoint.sh