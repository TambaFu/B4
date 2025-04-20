# 環境構築
実装言語: Python

## インストール
cuda118
ver:3.10.15

・Install Pytorch and torchvision
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```
・requirements.txtをインストール。しかし
#git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI&egg=pycocotoolsとする
```
pip install -r requirements.txt
```
・pycocoapiのダウンロード
```
git clone git+https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
```

・CUDA 演算子のコンパイル
```
cd models/dino/ops
python setup.py build install
```

・データセットのダウンロード
・COCO
http://images.cocodataset.org/zips/train2017.zip

http://images.cocodataset.org/zips/val2017.zip

http://images.cocodataset.org/annotations/annotations_trainval2017.zip

Please download [COCO 2017](https://cocodataset.org/) dataset and organize them as following:
```
COCODIR/
  ├── train2017/
  ├── val2017/
  └── annotations/
  	├── instances_train2017.json
  	└── instances_val2017.json
```

・VOC
http://host.robots.ox.ac.uk/pascal/VOC/
Pascal VOCに関しては、Pascal VOC 2007とPascal VOC 2012のtrainvalデータを用いて学習し、Pascal VOC2007のtestデータで評価する。Pascal VOC 2007のtrainvalデータは5011、testデータは4952、2012のtrainvalデータは11540ありますので、この方法での学習データの総数は16551、評価データの総数は4952、クラス数は20となります。すべてCOCOの形式になおす。

```
VOC/
  ├── train_images/
  ├── test_images/
  └── annotations/
  	├── voc_test_coco.json
  	└── voc_trainval_coco.json
```

・dataフォルダを作る
faster_vit_0.pth.tar
fastervit_0_224_1k.pth.tar
をhttps://huggingface.co/nvidia/FasterViT/tree/mainからダウンロードし、dataファイルに入れる