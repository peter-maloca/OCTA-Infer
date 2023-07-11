# OCTA Infer

Predict OCTA B-scans from OCT B-scans.

## Requirements
- Linux environment
- Python 3.7
- GPU support
  - Minimum 12 GB GPU RAM
  - CUDA v10.1
  - cuDNN v7

## Setup
- Install [CUDA toolchain](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) 
- Create virtual env
  ```bash
  <path/to/python3.7> -m venv <path/to/venv>
  source <path/to/venv>/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  ```

## Run Example
- Run OCT image through OCT-Infer net
  ```bash
  <path/to/python3.7> -m venv <path/to/venv>
  source <path/to/venv>/bin/activate
  cd <path/to/OCTA-Infer>/src
  python -m main
  ```
- Compare OCTA prediction (`data/images/octa_predicted`) with expected/recorded (`data/images/octa_expected`) 
