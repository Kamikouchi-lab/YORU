# YORU (Your Optimal Recognition Utility)

<img src="logos/YORU_logo.png" width="40%">
<img src="docs/imgs/title_movie.gif" width="50%">

“YORU” (Your Optimal Recognition Utility) is an open source animal behavior recognition system using Python. YORU can detect animal behaviors not only single-aminal behaviors but also social beahviors. YORU also provide online/offline analysis and closed-loop manipulation.


## Features

- TBA


# Quick install
1. Download or clone the YORU project.

    a. Download git

    ```
    conda install git
    ```

    b. Clone repository

    ```
    cd path¥to¥download
    git clone https://github.com/Kamikouchi-lab/YORU.git 
    ```

2. Install the GPU driver and [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit).

3. Create a virtual environment using [YORU.yml](YORU.yml) in command prompt or Anaconda prompt.
   
     ```
     conda env create -f "Peth/to/YORU.yml"
     ```

4. Activate the virtual environment in command prompt or Anaconda prompt.

     ```
     conda activate yoru
     ```
    
5. Install [Pytorch](https://pytorch.org) depending on the CUDA versions.

    - For CUDA==11.8

    ```
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

    - (torch, torchvision and torchaudio will be installed.)

6. Run YORU in command prompt or Anaconda prompt.

    ```
    conda activate yoru"
    cd "Peth/to/YORU/project/folder"
    python -m yoru
    ```


# Learn to YORU
- Learn step-by-step: [Tutorial](docs/overview.md)

- Learn by reading: TBA

# Requirements

## OS
- Windows 10 or later

## Hardware
- Memory: 16 GB or more

### Development environments
- OS: Windows 11
- CPU: Intel Core i9 (11th)
- GPU: NVIDIA RTX 3080
- Memory: DDR4 32 GB

# License:

TBA

# Third-Party Libraries and Licenses

This project includes code from the following repositories:

- [LabelImg](https://github.com/HumanSignal/labelImg): Licensed under the MIT License

- [yolov5](https://github.com/ultralytics/yolov5): Licensed under the AGPL-3.0 License