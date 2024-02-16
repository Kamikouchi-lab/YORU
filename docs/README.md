# READ-K

## 0.ドキュメントについて

このドキュメントは `@KeiPome` が個人用に作成したドキュメントであるため <b><span style="color:red;">非公式</span></b>です.

基本は頂いたdocをコピペですが、詰まるポイント等あれば加筆修正を行っています。

## 0.1環境構築

環境はAnacondaを使用して作成します。

- [Download](https://www.anaconda.com/download)
- [参考](https://www.python.jp/install/anaconda/)

## 1.install

YORUのインストール

1. GPUのドライバーと、[CUDA toolkit](https://developer.nvidia.com/cuda-toolkit), (CUDA 11.7 or 11.8が現状おすすめ)をインストールする

2. Anaconda promptでYORU.ymlを使って仮想環境を構築する
   - 下記をanaconda promptで実行する。
   
     ` $ conda env create -f "YORU.ymlのパス"` 
     

3. 仮想環境内でPyTorch(https://pytorch.org)のサイトからCUDAのバージョンに合うPyTorchのバージョンのインストールのコードを取得して、Anaconda promptに入力し、torchとその他(torchvisionと torchaudio)をインストールする。

   - CUDA==11.7 の場合
     `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`
   - CUDA==11.8 の場合
     `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`

4. YORUのレポジトリーを自分のパソコンにクローン、もしくはダウンロードする。

5. 仮想環境内でYORUを実行する
  - 下記をanaconda promptで実行する。

    ` $ cd yoru`
    ` $ python -m yoru` 


## 2.Training

1. Step1でプロジェクトを置きたいディレクトリを選び、プロジェクト名を入れ、Create YOLO projectを押す。

> - 指定したディレクトリにプロジェクト名のフォルダとその中に`config.yml`ができている。
> - すでにあるプロジェクトを読み込むときには、プロジェクトのディレクトリを選択して、Load YOLO projectを押す。

2. Step2のRun Grab GUIを起動し、動画からスクリーンショットを切り出す。
   Ⅰ. Grab GUIのVideo file pathで動画を選択する。
   Ⅱ. Save directoryを選ぶ。 (基本的にはプロジェクトフォルダのall_label_imagesが良いと思います。)
   Ⅲ. Frame nameを決める。
   Ⅳ. スクリーンショットを切り出す。
      ⅰ. Streaming movieで動画の再生
      ⅱ. 矢印キーで進む、戻る
      ⅲ. Grab Current FrameもしくはAltキーでフレームを保存する

3. Step 3のRun LabelImgを押し、LabelImgでラベル付けをする。

   -  LabelImg(https://github.com/HumanSignal/labelImg)

     > Save formatはYOLOで行う。 ViewタブのAuto Save modeをオンにするとやりやすい。


4. Labelした画像とtxtファイルを全てall_label_imagesにいれ、Step 4のMove Label Imagesをおす。
   
> trainとvalのフォルダにいい感じに画像とラベルが分かれます。)

5. Step 5でclasses.txtファイルを選択し、Add class info in YAML fileを押す。

> classes.txtの情報がconfig.ymlファイルに入力されます。)

6. YAML Pathでconfig.ymlファイルのパスが選択されていることを確認する。Training conditionを選択し、Train YOLOv5を押す。

> terminalでtrainingが開始されていれば良い。

## 3.Analysis 

1. Model PathでYOLOv5のモデルを選択する。

2. Movie Pathで解析したい動画を選択する。(複数選択可)

3. Result Directoryで結果を保存するディレクトリを指定する。

4. 動画を読み込むとpreviewに最初の動画が表示される。Flipなどを確認し、ある場合にはvertical flipとhorizontal flipで調整する。

5. YOLO analysisを押すと解析がスタートする。

​     	i. Create videosをチェックするとボックスの表示された動画が保存される。

​    	ii.  Tracking algorithmをチェックするとハンガリアン方によるIDがcsvに表示される。

