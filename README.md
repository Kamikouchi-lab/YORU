# YORU (Your Optimal Recognition Utility)

<img src="logos/YORU_logo.png" width="40%">
<img src="docs/imgs/title_movie.gif" width="50%">

[![Latest release](https://img.shields.io/github/v/release/Kamikouchi-lab/YORU?label=release)](https://github.com/Kamikouchi-lab/YORU/releases/latest)
[![Latest beta](https://img.shields.io/github/v/release/Kamikouchi-lab/YORU?include_prereleases&label=beta&color=orange)](https://github.com/Kamikouchi-lab/YORU/releases)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Documentation](https://img.shields.io/badge/docs-YORU-brightgreen.svg)](https://kamikouchi-lab.github.io/YORU_doc/)
[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-ff69b4?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/HMYamano)
[![GitHub stars](https://img.shields.io/github/stars/Kamikouchi-lab/YORU.svg?style=social&label=Star)](https://github.com/Kamikouchi-lab/YORU)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](https://github.com/Kamikouchi-lab/YORU/issues)

“YORU” (Your Optimal Recognition Utility) is an open-source animal behavior recognition system using Python. YORU can detect animal behaviors, not only single-animal behaviors but also social behaviors. YORU also provides online/offline analysis and closed-loop manipulation.

## Versions

| Channel | Version | Notes |
|---------|---------|-------|
| **Latest Release** | [v1.1.1](https://github.com/Kamikouchi-lab/YORU/releases/tag/v1.1.1) | Stable release recommended for general use |
| **Latest Beta** | [v2.0.0-beta.2](https://github.com/Kamikouchi-lab/YORU/releases/tag/v2.0.0-beta.2) | Preview of the next major version — may contain bugs |

> To use the beta version, check out the corresponding tag:
> ```
> git checkout v2.0.0-beta.2
> ```

## Features

- Comprehensive Behavior Detection: Recognizes both single-animal and social behaviors, and allows for user-defined animal appearances using deep learning techniques.

- Online/Offline Analysis: Supports real-time and post-experiment data analysis.

- Closed-Loop Manipulation: Enables interactive experiments with live feedback control.

- User-Friendly Interface: Provide the GUI-based software.

- Customizable: Allows you to customize various hardware manipulations in closed-loop system.

# Quick install

Follow these steps to install YORU quickly:

> These steps describe the conda route, which targets Windows (and Linux) with an NVIDIA GPU.
> On macOS, or if you already use [uv](https://docs.astral.sh/uv/), [Install via uv](#install-via-uv) below is simpler.

1. Download or clone the YORU project.
    ```
    cd "Path/to/download"
    git clone https://github.com/Kamikouchi-lab/YORU.git 
    ```

2. Install the appropriate GPU driver and the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit).

3. Create a virtual environment.

    Use [YORU.yml](https://github.com/Kamikouchi-lab/YORU/blob/main/YORU.yml) file to create a conda environment:
   
     ```
     conda env create -f "Path/to/YORU.yml"
     ```

    The environment file pins Python 3.10, which is the version YORU targets.

4. Activate the virtual environment in the command prompt or Anaconda prompt.

     ```
     conda activate yoru
     ```
    
5. Install [Pytorch](https://pytorch.org) corresponding to the CUDA versions.

    - For CUDA==11.8

    ```
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
    ```

   - For CUDA==12.1

    ```
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    ```
    

    - (torch, torchvision and torchaudio will be installed.)

6. Run YORU in the command prompt or Anaconda prompt.

    Navigate to the YORU project folder and execute:

    ```
    conda activate yoru
    cd "Path/to/YORU/project/folder"
    python -m yoru
    ```

7. etc.

    To check CUDA version in your environment:
    ```
    nvidia-smi
    ```
## Install via uv

The repository ships its own `pyproject.toml` and `uv.lock`, so [uv](https://docs.astral.sh/uv/) builds the whole environment in one step on Windows, Linux and macOS. uv also picks the right PyTorch build for your platform automatically: the CUDA wheels on Windows and Linux, and the MPS-enabled build from PyPI on macOS.

```
git clone https://github.com/Kamikouchi-lab/YORU.git
cd YORU
uv sync
uv run yoru
```

`uv run python -m yoru` does the same thing. There is no `uv init` step: the project is already initialised, and uv refuses to re-initialise a folder that already has a `pyproject.toml`.

## Compute device

YORU picks its compute device automatically, in this order: CUDA, then Apple MPS, then CPU. Nothing has to be configured for the usual cases.

To choose the device yourself, set the `YORU_DEVICE` environment variable before launching (`cuda`, `mps`, `cpu`, or a CUDA index such as `0`), or use the device selector in the training GUI:

```
YORU_DEVICE=cpu uv run yoru
```

On Windows, `set YORU_DEVICE=cpu` before the launch command. If the requested device is unavailable, YORU falls back to the next best one and writes a warning to `~/.yoru/logs/yoru.log` (`%USERPROFILE%\.yoru\logs\yoru.log` on Windows).

On Apple Silicon, MPS clearly helps training, but it is *not* faster than the CPU for single-frame realtime inference with the small YOLO models: we measured 60.8 FPS on MPS against 68.7 FPS on CPU for yolov8n at 640x480. For realtime detection on a Mac, `YORU_DEVICE=cpu` is worth trying.

The Faster R-CNN / Mask R-CNN / SSD models need torchvision 0.29 or newer to train on MPS at all; earlier releases diverge to an infinite loss within one epoch. The uv environment pins a new enough build, but the conda environment may not, so YORU checks the installed version and falls back to the CPU with a message when it is too old.

# Learn about YORU
- [User guides](https://kamikouchi-lab.github.io/YORU_doc/guides/01-install/)

- [Step-by-step Tutorial](https://kamikouchi-lab.github.io/YORU_doc/tutorial/01-preparation-tutorial/)

- [Testing Guide](https://kamikouchi-lab.github.io/YORU_doc/devnotes/yoru-test/)

# Requirements

## OS
- Windows 10 or later, with an NVIDIA GPU. This is the primary target, and the only platform tested end to end including the closed-loop hardware.
- Linux, with an NVIDIA GPU and CUDA. It uses the same CUDA wheels as Windows, but has seen much less testing.
- macOS 14 (Sonoma) or later, on Apple Silicon. The floor comes from PyTorch: torchvision 0.29 is the first release whose detection models train correctly on MPS, and it requires a torch build whose macOS wheels target 14.0.

## Hardware
- Memory: 16 GB RAM or more
- GPU: NVIDIA GPU compatible with the required CUDA version, or an Apple Silicon (M-series) Mac, which uses MPS. YORU also runs on the CPU alone, but detection is much slower.

### Development environments
- OS: Windows 11
- CPU: Intel Core i9 (11th)
- GPU: NVIDIA RTX 3080
- Memory: DDR4 32 GB

## Software
- Python 3.10

### Platform notes
- On macOS, training and inference are verified end to end, but the GUI has had far less testing there than on Windows.
- The closed-loop hardware manipulation depends on drivers we only use on Windows: NI-DAQ (via `nidaqmx`) needs the NI-DAQmx driver, and the Arduino path is untested elsewhere. Detection, training and analysis do not need any of it.

# Reference
 - Yamanouchi, H. M., Takeuchi, R. F., Chiba, N., Hashimoto, K., Shimizu, T., Osakada, F., Tanaka, R., & Kamikouchi, A. (2026). YORU: Animal behavior detection with object-based approach for real-time closed-loop feedback. *Science Advances*, 12(7). https://doi.org/10.1126/sciadv.adw2109



# License:

AGPL-3.0 License:  YORU is intended for research/academic/personal use only. See the [LICENSE](LICENSE) file for more details.

# Third-Party Libraries and Licenses

This project includes code from the following repositories:

- [LabelImg](https://github.com/HumanSignal/labelImg): Licensed under the MIT License

- [yolov5](https://github.com/ultralytics/yolov5): Licensed under the AGPL-3.0 License

