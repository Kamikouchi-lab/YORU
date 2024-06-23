# YORU (Your Optimal Recognition Utility)

<img src="logos/YORU_logo.png" width="40%">

“YORU” (Your Optimal Recognition Utility) is an open source animal behavior recognition system using Python. YORU can detect animal behaviors not only single-aminal behaviors but also social beahviors. YORU also provide online/offline analysis and closed-loop manipulation.

### Features

- TBA


## Quick install
1. Download or clone the YORU project.

2. Install the GPU driver and [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit).

3. Create a virtual environment using [YORU.yml](YORU.yml) in command prompt or Anaconda prompt.
   
     `conda env create -f "Peth/to/YORU.yml"`

4. Activate the virtual environment in command prompt or Anaconda prompt.

     `conda activate yoru`

5. Install [Pytorch](https://pytorch.org) depending on the CUDA versions.

    - For CUDA==11.8

    `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
    - (torch, torchvision and torchaudio will be installed.)

6. Run YORU in command prompt or Anaconda prompt.

    `conda activate yoru"`

    `cd "Peth/to/YORU/project/folder"`
    
    `python -m yoru`


## Learn to YORU
- Learn step-by-step: [Tutorial](docs/overview.md)

- Learn by reading: TBA

# License:

TBA

