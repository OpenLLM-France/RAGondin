FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Définir les variables d'environnement pour éviter les interactions
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC


RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    gnupg \
    && add-apt-repository ppa:deadsnakes/ppa \
    && rm -rf /var/lib/apt/lists/*



WORKDIR /service


# Vérifier et installer libcudnn8 si nécessaire
RUN apt-get update && apt-get install -y wget && \
    if ! dpkg -l | grep -q libcudnn8; then \
        apt-get update && apt-get install -y libcudnn8; \
    fi

RUN apt update
RUN apt install -y curl

#FROM python:3.12.7-alpine3.20
#WORKDIR /app

#RUN apk --no-cache add curl
# Installer Poetry




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
#RUN apt-get update && apt-get install -y \
#    python3.12 \
#    python3.12-venv \
#    python3.12-dev
#COPY pyproject.toml /service/
#COPY poetry.lock /service/
# Installer Poetry
RUN curl -sSL https://install.python-poetry.org | python3.12 -
RUN    export PATH="/root/.local/bin:$PATH" 
ENV PATH="/root/.local/bin:$PATH"

#RUN poetry env use python3.12 && \
#    poetry install --no-interaction -vv --no-root


