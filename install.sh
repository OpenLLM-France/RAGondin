#!/bin/bash

#  libcudnn8 pour cuda 
apt-get update
apt-get install -y libcudnn8 curl

curl -sSL https://install.python-poetry.org | python3 -

export PATH="$HOME/.poetry/bin:$PATH"

poetry install