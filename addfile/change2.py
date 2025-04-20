#from pycocotools.coco import COCO
#
## 学習用データの確認
#coco_trainval = COCO("voc_trainval_coco.json")
#print(f"学習データの画像数: {len(coco_trainval.imgs)}")
#print(f"学習データのアノテーション数: {len(coco_trainval.anns)}")
#
## 評価用データの確認
#coco_test = COCO("voc_test_coco.json")
#print(f"評価データの画像数: {len(coco_test.imgs)}")
#print(f"評価データのアノテーション数: {len(coco_test.anns)}")


import os
import shutil

def merge_trainval_and_split_images(voc_dirs, output_dir, trainval_files, test_file):
    """
    Pascal VOC 2007 と 2012 の trainval を統合し、画像を分割
    """
    # 画像フォルダ
    images_dir_2007 = os.path.join(voc_dirs[0], "JPEGImages")
    images_dir_2012 = os.path.join(voc_dirs[1], "JPEGImages")

    # 出力フォルダ
    train_images_dir = os.path.join(output_dir, "train_images")
    test_images_dir = os.path.join(output_dir, "test_images")

    # 出力フォルダ作成
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)

    # trainval を統合
    trainval_list = []
    for trainval_file, images_dir in zip(trainval_files, [images_dir_2007, images_dir_2012]):
        with open(trainval_file, "r") as f:
            trainval_list.extend([(line.strip(), images_dir) for line in f.readlines()])

    # trainval データを train 用フォルダにコピー
    for img_name, img_dir in trainval_list:
        img_file = os.path.join(img_dir, f"{img_name}.jpg")
        if os.path.exists(img_file):
            shutil.copy(img_file, os.path.join(train_images_dir, f"{img_name}.jpg"))

    print(f"Train データ統合完了: {len(trainval_list)} 画像")

    # test データを test 用フォルダにコピー
    with open(test_file, "r") as f:
        test_list = [line.strip() for line in f.readlines()]

    for img_name in test_list:
        img_file = os.path.join(images_dir_2007, f"{img_name}.jpg")
        if os.path.exists(img_file):
            shutil.copy(img_file, os.path.join(test_images_dir, f"{img_name}.jpg"))

    print(f"Test データ分割完了: {len(test_list)} 画像")
    print(f"Train ディレクトリ: {train_images_dir}")
    print(f"Test ディレクトリ: {test_images_dir}")

# 使用例
voc_dirs = ["VOCdevkit/VOC2007", "VOCdevkit/VOC2012"]  # Pascal VOC の 2007 と 2012 のディレクトリ
output_dir = "VOC_split"                              # 出力先ディレクトリ
trainval_files = [
    "VOCdevkit/VOC2007/ImageSets/Main/trainval.txt",  # 2007 の trainval
    "VOCdevkit/VOC2012/ImageSets/Main/trainval.txt",  # 2012 の trainval
]
test_file = "VOCdevkit/VOC2007/ImageSets/Main/test.txt"  # 2007 の test データ

merge_trainval_and_split_images(voc_dirs, output_dir, trainval_files, test_file)
