#!/bin/bash

# install torch
pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# install requirements
pip install -r requirements.txt
pip install -r requirements_dev.txt