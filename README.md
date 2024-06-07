# YORU (Your Optimal Recognition Utility)
Online/Offline tracking analysis and closed-loop manipulation tool

画像

### Features

- TBA


## Quick install
1. Install GPU driver and [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit)

2. Create virtual invironment using [YORU.yml](YORU.yml) in command prompt or Anaconda prompt
   
     ` $ conda env create -f "YORU.ymlのパス"`

4. 

     

3. 仮想環境内でPyTorch(https://pytorch.org)のサイトからCUDAのバージョンに合うPyTorchのバージョンのインストールのコードを取得して、Anaconda promptに入力し、torchとその他(torchvisionと torchaudio)をインストールする。

   - CUDA==11.7 の場合
     `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`
   - CUDA==11.8 の場合
     `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`

4. 仮想環境内でYORUを実行する
  - 下記をanaconda promptで実行する。

     ` $ python -m yoru`

## Learn to YORU
- Learn step-by-step: Tutorial(TBA)

- Learn by reading: TBA

# License:

TBA

