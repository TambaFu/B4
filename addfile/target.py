
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

# COCOアノテーションファイルのパス
annotation_path = "bird/annotation/coco_annotations_train.json"  # JSONのパス
image_path = "bird/traindata/026.png"  # 画像ファイルのパス

# アノテーションデータをロード
with open(annotation_path, "r") as f:
    coco_data = json.load(f)

# 画像IDを取得 (file_name が "026.png" の image_id を探す)
image_id = None
for image in coco_data["images"]:
    if image["file_name"] == "026.png":
        image_id = image["id"]
        break

if image_id is None:
    raise ValueError("指定された画像 '026.png' の情報が見つかりません。")

# image_id に対応するアノテーションを抽出
annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]

# COCOのカテゴリーIDとラベル名のマッピング（仮のマッピング）
category_id_to_name = {1: "bird"}  # 必要に応じて変更

# 画像を読み込む（OpenCVを使用）
image = cv2.imread(image_path)
if image is None:
    raise ValueError("画像ファイルが見つかりません。")

# OpenCVのBGRをRGBに変換
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 画像をプロット
fig, ax = plt.subplots(1, figsize=(10, 6))
ax.imshow(image)

# バウンディングボックスを描画（IDを排除し、ラベル名のみ）
for ann in annotations:
    x, y, w, h = ann["bbox"]
    category_name = category_id_to_name.get(ann["category_id"], "unknown")  # カテゴリー名取得
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="red", facecolor="none")
    ax.add_patch(rect)
    ax.text(x, y - 5, category_name, color="red", fontsize=5, bbox=dict(facecolor="white", alpha=0.5))

# タイトルと表示
ax.set_title("Bounding Boxes with Labels on 026.png")
plt.axis("off")
plt.show()