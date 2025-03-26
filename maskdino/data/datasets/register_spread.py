# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import numpy as np
import os
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from detectron2.utils.file_io import PathManager

BASE_DIR = '/mnt/raid1/dataset/'
#BASE_DIR = '/mnt/data/dataset'
#ANNO_DIR = 'spread/coco-annotations/spread-v2-coco'
#ANNO_DIR = 'spread/coco-annotations/spread-coco-annotations-o1-36170-samples-612023-instances-20250314'
#ANNO_DIR = 'spread/coco-annotations/spread-coco-annotations-o1-36173-samples-961624-instances-everything-at-960x540px-20250319'
#ANNO_DIR = 'spread/coco-annotations/spread-coco-annotations-o1-36173-samples-611442-instances-masks-480x270-polygon-annotations-20250320'
ANNO_DIR = 'spread/coco-annotations/spread-coco-annotations-o1-36173-samples-611442-instances-masks-480x270-polygon-annotations-20250320-renamed-base-dir-to-spread-480x270'
ANNO_DIR = 'spread/coco-annotations/spread-coco-annotations-o1-36173-samples-961624-poly-annotations-960x540-20250325'

SPREAD_CATEGORIES = [{"id": 1, "name": "tree", "supercategory": "plant"},]

def _get_spread_instances_meta():
	thing_ids     = [k["id"]   for k in SPREAD_CATEGORIES]
	thing_classes = [k["name"] for k in SPREAD_CATEGORIES]

	thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
	ret = {
		"thing_dataset_id_to_contiguous_id"	: thing_dataset_id_to_contiguous_id,
		"thing_classes"				: thing_classes,
		"ignore_label"				: 255
	}
	return ret

def register_all_spread(spread_dir):
	register_coco_instances("spread_train", _get_spread_instances_meta(), f"{spread_dir}/train.json", BASE_DIR + ANNO_DIR)
	register_coco_instances("spread_valid", _get_spread_instances_meta(), f"{spread_dir}/valid.json", BASE_DIR + ANNO_DIR)
	register_coco_instances("spread_test" , _get_spread_instances_meta(), f"{spread_dir}/test.json" , BASE_DIR + ANNO_DIR)

#SPREAD_ANNO_DIR = '/mnt/raid1/dataset/spread/spread-v2'
#SPREAD_ANNO_DIR = '/mnt/data/dataset/spread/spread-v2'
#SPREAD_ANNO_DIR = BASE_DIR + '/spread/spread-v2'
SPREAD_ANNO_DIR = BASE_DIR + ANNO_DIR
#_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_spread(SPREAD_ANNO_DIR)
