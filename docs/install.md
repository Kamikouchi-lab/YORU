# Install

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
    
    >(torch, torchvision and torchaudio will be installed.)

6. Run YORU in command prompt or Anaconda prompt.

    ```
    conda activate yoru"
    cd "Peth/to/YORU/project/folder"
    python -m yoru
    ```