# Install

> **Note:** From v2.0 the launcher uses [pywebview](https://pywebview.flowrl.com/)
> (the native OS WebView), so no external browser and no local web server are
> required. Earlier versions needed Google Chrome for the `eel` package.

1. Check the installation of [Miniconda](https://docs.anaconda.com/miniconda/)

> Anaconda's [TERMS OF SERVICE](https://legal.anaconda.com/policies/en?name=terms-of-service#terms-of-service) was changed. If you used Anaconda in an organization that has two hundred (200) or more employees or contractors, you have to be careful.

> Currently, you can use miniconda freely.

3. Download or clone the YORU project.

    a. Download git

    ```
    conda install git
    ```

    b. Clone repository

    ```
    cd "Path/to/download"
    git clone https://github.com/Kamikouchi-lab/YORU.git 
    ```

4. Install the GPU driver and [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit).

5. Create a virtual environment using [YORU.yml](../YORU.yml) in command prompt or Anaconda prompt.
   
     ```
     conda env create -f "Path/to/YORU.yml"
     ```

6. Activate the virtual environment in command prompt or miniconda prompt.

     ```
     conda activate yoru
     ```

7. Install [Pytorch](https://pytorch.org) for your GPU.

    - **RTX 50-series (Blackwell) and every card before it — CUDA 12.8**

    ```
    pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
    ```

    > This is the build YORU is developed against, and the one the `uv`
    > install below resolves to. Its wheels carry `sm_61` through `sm_120`, so
    > it covers a GTX 10-series and an RTX 5090 alike. It needs driver 570 or
    > newer (Windows 572.xx); check yours with `nvidia-smi`.

    - **Older cards on an older driver — CUDA 12.6**

    ```
    pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
    ```

    > Only if the driver cannot be updated. This build stops at `sm_90`, so on
    > an RTX 50-series card every CUDA call fails with *"no kernel image is
    > available for execution on the device"* even though `torch.cuda
    > .is_available()` returns True.

    Check that the card is actually usable, not merely detected:

    ```
    python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list()); print(torch.zeros(1).cuda() + 1)"
    ```

    > The printed architecture list must contain the one your GPU needs —
    > `sm_120` for the RTX 50-series, `sm_89` for the 40-series, `sm_86` for
    > the 30-series. `torch.cuda.is_available()` alone does **not** tell you
    > this: it returns True on a card the build has no kernels for.

    > **Python version.** YORU runs on Python 3.9, and torch 2.8.0 is the last
    > release built for it — torch 2.9 and newer need Python 3.10+. There is no
    > newer torch to move to without moving Python first.

    > torchaudio is not used by YORU and does not need to be installed.

8. Run YORU in a command prompt or miniconda prompt.

    ```
    conda activate yoru
    cd "Path/to/YORU/project/folder"
    python -m yoru
    ```

## Alternative: install with uv

[uv](https://docs.astral.sh/uv/) resolves everything from `pyproject.toml` /
`uv.lock`, so steps 5-7 are not needed:

```
cd "Path/to/YORU"
uv sync
uv run python -m yoru
```
