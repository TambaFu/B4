# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco
from .coco import build2 as build_voc
from .coco import build3 as build_bird
from .coco import build4 as build_bird2
def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'bird':
        return build_bird(image_set, args)
    if args.dataset_file == 'bird2':
        return build_bird2(image_set, args)
    if args.dataset_file == 'voc':
        return build_voc(image_set, args)
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'o365':
        from .o365 import build_o365_combine
        return build_o365_combine(image_set, args)
    if args.dataset_file == 'vanke':
        from .vanke import build_vanke
        return build_vanke(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
