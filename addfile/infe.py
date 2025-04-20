import os, sys
import torch, json
import numpy as np

from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops

model_config_path = "~/Documents/DINO/config/DINO/DINO_4scale_faster_vit_4_21k_224.py" # change the path of the model config file
model_checkpoint_path = "DINO_4scale_faster_vit_4_21k_224_ms_coco.pth" # change the path of the model checkpoint
# See our Model Zoo section in README.md for more details about our pretrained models.
#config ファイルに含まれる情報（例: 学習率、エポック数、モデル構造など）が辞書形式で args へ
args = SLConfig.fromfile(model_config_path) 
args.device = 'cuda' 
#print(args)
#関数呼び出し: build_model_mainが呼び出されると、引数argsを受け取ります。
#モデル名の確認: argsに指定されたモデル名がサポートされているかどうかを確認します。
#モデル構築関数の取得: サポートされているモデル名に基づいて、モデルを構築するための関数を取得します。
#モデルの構築: 取得した関数を使ってモデル、損失関数、後処理関数を構築します。
#結果の返却: 構築されたモデルとその設定を返します。
#model: DINOモデル本体（PyTorchのnn.Module形式）。
#criterion: 損失関数（学習時に利用）。
#postprocessors: 推論結果を後処理する関数群。
model, criterion, postprocessors = build_model_main(args)
#print(model)
print(postprocessors)
#学習済みモデルの重みをロード。
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
#モデルに学習済みの重みを適用。
model.load_state_dict(checkpoint['model'])
_ = model.eval()

# load coco names
#JSON形式のファイルを読み込んで、キーを整数に変換した辞書を作成する動作をしている
with open('util/coco_id2name.json') as f:
    id2name = json.load(f)
    id2name = {int(k):v for k,v in id2name.items()}
#args（辞書オブジェクト）に新しいキーと値を追加
args.dataset_file = 'coco'
args.coco_path = "COCOIR" # the path of coco
args.fix_size = False

dataset_val = build_dataset(image_set='val', args=args)
image, targets = dataset_val[0]
#検証用データセットとして、すべての検証画像とアノテーションがロードされます。
#各データ項目にアクセスするには、インデックスを使用します（例: dataset_val[0]
#print(dataset_val)
#Rチャネル、Gチャネル、Bチャネルには、それぞれ画像の画素数分の要素が含まれている
#print(image)
#targets は、モデルが推論結果を評価する際に使う正解データ
#print(targets)
# build gt_dict for vis
'''
box_label = [id2name[int(item)] for item in targets['labels']]
print(box_label)
gt_dict = {
    'boxes': targets['boxes'],
    'image_id': targets['image_id'],
    'size': targets['size'],
    'box_label': box_label,
}
#COCOVisualizerのvisualizeでコンソールにバウンディングボックスで可視化したものを表示
#matplotlibで表示部分を実現
#ちなみにこれは正解データを表示している
import numpy as np

# 保存先のファイル名
txt_file = "output_image.txt"

# img を CPU 上の NumPy 配列に変換（そのままテンソルでも可）
img_array = image.cpu().numpy()  # img は (3, H, W) のテンソル形式を仮定

# ファイルに書き込み
with open(txt_file, "w") as f:
    # 配列を文字列に変換して書き込む
    f.write(str(img_array))

print(f"Image tensor has been saved as a text file to {txt_file}")
vslzr = COCOVisualizer()
vslzr.visualize(image, gt_dict, savedir=None)
#output に検出結果が格納されます（例: 各物体のバウンディングボックス、スコア、ラベルなど）。
#image[None]:
#image（単一画像）をバッチ形式に変換します（次元を1つ増やす）。具体的には、形状が [C, H, W] から [1, C, H, W] になります。
output = model.cuda()(image[None].cuda())
#モデルの出力に後処理を適用して、実際の画像スケールに対応する形式に変換します。
output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
#0.3: 物体の検出信頼度が30%以上の結果を採用
thershold = 0.3 # set a thershold

scores = output['scores']
labels = output['labels']
#検出された物体のバウンディングボックス（[xmin, ymin, xmax, ymax] 形式）。
# このコードでは、box_ops.box_xyxy_to_cxcywh を使用して、中心形式（[cx, cy, w, h]）に変換しています。
boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
select_mask = scores > thershold

box_label = [id2name[int(item)] for item in labels[select_mask]]
print(box_label)
pred_dict = {
    'boxes': boxes[select_mask],
    'size': targets['size'],
    'box_label': box_label
}
vslzr.visualize(image, pred_dict, savedir=None)'''
from PIL import Image
import datasets.transforms as T

image = Image.open("bird/evaldata/035.png").convert("RGB") # load image

# transform images
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image, _ = transform(image, None)

# predict images
output = model.cuda()(image[None].cuda())
output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

thershold = 0.3 # set a thershold

vslzr = COCOVisualizer()

scores = output['scores']
labels = output['labels']
boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
select_mask = scores > thershold

box_label = [id2name[int(item)] for item in labels[select_mask]]
print(box_label)
pred_dict = {
    'boxes': boxes[select_mask],
    'size': torch.Tensor([image.shape[1], image.shape[2]]),
    'box_label': box_label
}
vslzr.visualize(image, pred_dict, savedir=None, dpi=100)

