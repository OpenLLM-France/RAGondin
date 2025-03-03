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

# Set workdir inside the container


# Copy only necessary files to reduce image size
# Set the workdir early in the Dockerfile
WORKDIR /ragondin

# Install uv & setup venv
COPY pyproject.toml uv.lock /ragondin/
RUN pip3 install uv && \
    uv sync

# Copy source code
COPY ragondin/ .

# Copy assests & config
COPY public/ /ragondin/public/
COPY .hydra_config/ ../.hydra_config/
ENV PYTHONPATH=/ragondin/


