# Install

YORU can be installed in two ways:

- **[Install with uv](#install-with-uv)** -- one command builds the whole
  environment, including Python itself, on Windows, Linux and macOS. This is the
  only supported route on macOS.
- **[Install with conda](#install-with-conda)** -- the original route, for
  Windows and Linux with an NVIDIA GPU.

Whichever you pick, read [Prerequisites](#prerequisites) first: those pieces
have to come from outside the Python environment.

# Prerequisites

## 1. No browser needed (from v2.0)

The YORU launcher uses [pywebview](https://pywebview.flowrl.com/), the native
OS WebView, so **no external browser and no local web server are required**.
Earlier versions served the launcher through
[Eel](https://github.com/python-eel/Eel) and needed Google Chrome or Chromium;
that requirement is gone. The training, evaluation, analysis and real-time
windows are native windows, as before.

## 2. An NVIDIA driver, to use a GPU (Windows / Linux)

Install the GPU driver. It has to support CUDA 12.x: **527.41 or newer on
Windows, 525.60.13 or newer on Linux**. Check the installed version with:

```
nvidia-smi
```

The PyTorch wheels carry their own CUDA runtime -- `cudart`, cuBLAS and cuDNN
are shipped inside the `torch` package itself -- so the
[CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) is **not required by
either route**. Install it only if you need `nvcc` to compile CUDA extensions of
your own, or want the profiling tools.

The `CUDA Version` reported by `nvidia-smi` is the highest your driver supports,
not the version in use: a driver reporting 13.x runs the CUDA 12.4 wheels
without a toolkit installed.

Without a usable GPU, YORU runs on the CPU. Everything works, just more slowly.

## 3. macOS extras (Apple Silicon)

- YORU needs **macOS 14 (Sonoma) or later on Apple Silicon**. Intel Macs are not
  supported, because the macOS wheels of the PyTorch and Qt versions YORU pins
  are arm64 only.
- Install the **Xcode Command Line Tools** before `uv sync`. `imgui` ships no
  arm64 wheel and is compiled during the sync:

  ```
  xcode-select --install
  ```

- macOS asks for permission the first time YORU uses a camera
  (**Camera**), the key-press triggers (**Input Monitoring** /
  **Accessibility**, via `pynput`) or screen capture (**Screen Recording**, via
  `mss`). The prompts target the terminal application you launched YORU from.
  Grant them in *System Settings > Privacy & Security*, then relaunch.
- The NI-DAQ closed-loop path needs a driver NI ships for Windows only, so it is
  unavailable on macOS.

# Install with uv

[uv](https://docs.astral.sh/uv/) reads the `pyproject.toml` and `uv.lock` in the
repository and builds the environment from them, picking the right PyTorch for
your platform: the CUDA wheels on Windows and Linux, the MPS-enabled build from
PyPI on macOS. No system Python and no conda are needed -- uv downloads the
Python 3.10 the project asks for.

1. Install uv.

    Windows (PowerShell):

    ```
    winget install --id=astral-sh.uv -e
    ```

    macOS / Linux:

    ```
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    `brew install uv` works on macOS too, and the
    [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
    lists the standalone installers. Open a new terminal afterwards so that `uv`
    is on `PATH`.

2. Download or clone the YORU project.

    ```
    cd "Path/to/download"
    git clone https://github.com/Kamikouchi-lab/YORU.git
    ```

    Downloading the ZIP from GitHub works just as well if you do not have git.

3. Build the environment.

    ```
    cd YORU
    uv sync
    ```

    The first sync downloads PyTorch and takes a few minutes.

4. Run YORU **from the repository root** -- the launcher resolves `web/` and
   `config/` relative to the working directory.

    ```
    uv run yoru
    ```

    `uv run python -m yoru` does the same thing.

To confirm that the GPU was picked up:

```
uv run python -c "import torch; print(torch.cuda.is_available())"
```

# Install with conda

1. Check the installation of [Miniconda](https://docs.anaconda.com/miniconda/)

> Anaconda's [TERMS OF SERVICE](https://legal.anaconda.com/policies/en?name=terms-of-service#terms-of-service) was changed. If you used Anaconda in an organization that has two hundred (200) or more employees or contractors, you have to be careful.

> Currently, you can use miniconda freely.

2. Download or clone the YORU project.

    a. Download git

    ```
    conda install git
    ```

    b. Clone repository

    ```
    cd "Path/to/download"
    git clone https://github.com/Kamikouchi-lab/YORU.git 
    ```

3. Install the GPU driver. The [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) is not needed -- the wheels in step 6 carry their own CUDA runtime -- so the `cu118` / `cu121` choice there only has to be one your driver supports.

4. Create a virtual environment using [YORU.yml](../YORU.yml) in command prompt or Anaconda prompt.
   
     ```
     conda env create -f "Path/to/YORU.yml"
     ```

5. Activate the virtual environment in command prompt or miniconda prompt.

     ```
     conda activate yoru
     ```

6. Install [Pytorch](https://pytorch.org) corresponding to the CUDA versions.

    - For CUDA==11.8

    ```
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
    ```

   - For CUDA==12.1

    ```
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    ```
    
    >(torch, torchvision and torchaudio will be installed.)

7. Run YORU in a command prompt or miniconda prompt.

    ```
    conda activate yoru
    cd "Path/to/YORU/project/folder"
    python -m yoru
    ```

# Choosing the compute device

YORU picks CUDA, then Apple MPS, then the CPU. To choose yourself, set
`YORU_DEVICE` (`cuda`, `mps`, `cpu`, or a CUDA index such as `0`) before
launching, or use the device selector in the training GUI. An unavailable
device falls back to the next best one, with a warning in
`~/.yoru/logs/yoru.log` (`%USERPROFILE%\.yoru\logs\yoru.log` on Windows).

`YORU_DEVICE` covers YOLOv8, YOLO11, RT-DETR and the torchvision detectors.
