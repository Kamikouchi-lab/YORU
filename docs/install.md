# Install

0. Check the instllation of [Google Chrome](https://www.google.com/intl/ja/chrome/)

- eel package need to use Google Chrome.

1. Check the instllation of [Miniconda](https://docs.anaconda.com/miniconda/)

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

5. Create a virtual environment using [YORU.yml](YORU.yml) in command prompt or Anaconda prompt.
   
     ```
     conda env create -f "Path/to/YORU.yml"
     ```

6. Activate the virtual environment in command prompt or miniconda prompt.

     ```
     conda activate yoru
     ```

7. Install [Pytorch](https://pytorch.org) depending on the CUDA versions.

    - For CUDA==11.8

    ```
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
    
    >(torch, torchvision and torchaudio will be installed.)

8. Run YORU in a command prompt or miniconda prompt.

    ```
    conda activate yoru
    cd "Path/to/YORU/project/folder"
    python -m yoru
    ```
