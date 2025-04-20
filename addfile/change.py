import os
import json
import xml.etree.ElementTree as ET

def parse_voc_xml(xml_file, image_id, annotation_id, label_map):
    """Pascal VOC の XML ファイルを解析して COCO アノテーション形式を生成"""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 画像情報を取得
    file_name = root.find("filename").text
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height,
    }

    annotations = []
    for obj in root.findall("object"):
        category_name = obj.find("name").text
        if category_name not in label_map:
            continue
        category_id = label_map.index(category_name) + 1
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        width_box = xmax - xmin
        height_box = ymax - ymin
        annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [xmin, ymin, width_box, height_box],
            "area": width_box * height_box,
            "iscrowd": 0,
        })
        annotation_id += 1

    return image_info, annotations, annotation_id

def voc_to_coco(voc_dirs, output_json, label_map, split_files):
    """Pascal VOC を COCO フォーマットに変換"""
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(label_map)],
    }

    image_id = 1
    annotation_id = 1

    for voc_dir, split_file in zip(voc_dirs, split_files):
        annotations_dir = os.path.join(voc_dir, "Annotations")
        images_dir = os.path.join(voc_dir, "JPEGImages")
        with open(split_file, "r") as f:
            image_list = [line.strip() for line in f.readlines()]

        for img_name in image_list:
            xml_file = os.path.join(annotations_dir, f"{img_name}.xml")
            if not os.path.exists(xml_file):
                continue
            image_info, annotations, annotation_id = parse_voc_xml(
                xml_file, image_id, annotation_id, label_map
            )
            coco_format["images"].append(image_info)
            coco_format["annotations"].extend(annotations)
            image_id += 1

    # JSON ファイルに保存
    with open(output_json, "w") as f:
        json.dump(coco_format, f, indent=4)
    print(f"COCOフォーマットを保存しました: {output_json}")

# 使用例
voc_dirs = ["VOCdevkit/VOC2007", "VOCdevkit/VOC2012"]
split_files = [
    "VOCdevkit/VOC2007/ImageSets/Main/trainval.txt",
    "VOCdevkit/VOC2012/ImageSets/Main/trainval.txt",
]
output_json = "voc_trainval_coco.json"
label_map = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

voc_to_coco(voc_dirs, output_json, label_map, split_files)

voc_dirs = ["VOCdevkit/VOC2007"]
split_files = ["VOCdevkit/VOC2007/ImageSets/Main/test.txt"]
output_json = "voc_test_coco.json"

voc_to_coco(voc_dirs, output_json, label_map, split_files)

