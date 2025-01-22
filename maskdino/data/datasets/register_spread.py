# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import numpy as np
import os
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from detectron2.utils.file_io import PathManager

SPREAD_CATEGORIES = [{"id": 1, "name": "tree", "supercategory": "plant"},]

def _get_spread_instances_meta():
	thing_ids     = [k["id"]   for k in SPREAD_CATEGORIES]
	thing_classes = [k["name"] for k in SPREAD_CATEGORIES]

	thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
	'''
	assert len(thing_ids) == 100, len(thing_ids)
	# Mapping from the incontiguous ADE category id to an id in [0, 99]
	thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
	thing_classes = [k["name"] for k in SPREAD_CATEGORIES]
	ret = {
		"thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
		"thing_classes": thing_classes,
	}
	return ret
	'''
	ret = {
		"thing_dataset_id_to_contiguous_id"	: thing_dataset_id_to_contiguous_id,
		"thing_classes"				: thing_classes,
		"ignore_label"				: 255
	}
	return ret

def register_all_spread(spread_dir):
	register_coco_instances("spread_train", _get_spread_instances_meta(), f"{spread_dir}/train.json", "/mnt/raid1/dataset/spread/spread-v2-coco")
	register_coco_instances("spread_valid", _get_spread_instances_meta(), f"{spread_dir}/valid.json", "/mnt/raid1/dataset/spread/spread-v2-coco")
	register_coco_instances("spread_test" , _get_spread_instances_meta(), f"{spread_dir}/test.json" , "/mnt/raid1/dataset/spread/spread-v2-coco")

'''
def _get_ade_instances_meta():
    thing_ids = [k["id"] for k in ADE_CATEGORIES]
    assert len(thing_ids) == 100, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in ADE_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def register_all_ade20k_instance(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_ade_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
'''



SPREAD_DIR = '/mnt/raid1/dataset/spread/spread-v2'
#_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_spread(SPREAD_DIR)
