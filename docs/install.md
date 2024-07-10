# Install

0. Check the instlation of [Google Chrome](https://www.google.com/intl/ja/chrome/)

- eel package need to use Google Chrome.

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

3. Install the GPU driver and [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit).

4. Create a virtual environment using [YORU.yml](YORU.yml) in command prompt or Anaconda prompt.
   
     ```
     conda env create -f "Peth/to/YORU.yml"
     ```

5. Activate the virtual environment in command prompt or Anaconda prompt.

     ```
     conda activate yoru
     ```

6. Install [Pytorch](https://pytorch.org) depending on the CUDA versions.

    - For CUDA==11.8

    ```
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
    
    >(torch, torchvision and torchaudio will be installed.)

7. Run YORU in command prompt or Anaconda prompt.

    ```
    conda activate yoru
    cd "Peth/to/YORU/project/folder"
    python -m yoru
    ```
