# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

CAMVID_CATEGORIES = [
    {"color": [64,128,64], "isthing": 1, "id": 1, "name": "Animal"},
    {"color": [192,0,128], "isthing": 1, "id": 2, "name": "Archway"},
    {"color": [0,128, 192], "isthing": 1, "id": 3, "name": "Bicyclist"},
    {"color": [0, 128, 64], "isthing": 1, "id": 4, "name": "Bridge"},
    {"color": [128, 0, 0], "isthing": 1, "id": 5, "name": "Building"},
    {"color": [64, 0, 128], "isthing": 1, "id": 6, "name": "Car"},
    {"color": [64, 0, 192], "isthing": 1, "id": 7, "name": "CartLuggagePram"},
    {"color": [192, 128, 64], "isthing": 1, "id": 8, "name": "Child"},
    {"color": [192, 192, 128], "isthing": 1, "id": 9, "name": "Column_Pole"},
    {"color": [64, 64, 128], "isthing": 1, "id": 10, "name": "Fence"},
    {"color": [128, 0, 192], "isthing": 1, "id": 11, "name": "LaneMkgsDriv"},
    {"color": [192, 0, 64], "isthing": 1, "id": 12, "name": "LaneMkgsNonDriv"},
    {"color": [128, 128, 64], "isthing": 1, "id": 13, "name": "Misc_Text"},
    {"color": [192, 0, 192], "isthing": 1, "id": 14, "name": "MotorcycleScooter"},
    {"color": [128, 64, 64], "isthing": 1, "id": 15, "name": "OtherMoving"},
    {"color": [64, 192, 128], "isthing": 1, "id": 16, "name": "ParkingBlock"},
    {"color": [64, 64, 0], "isthing": 1, "id": 17, "name": "Pedestrian"},
    {"color": [128, 64, 128], "isthing": 1, "id": 18, "name": "Road"},
    {"color": [128, 128, 192], "isthing": 1, "id": 19, "name": "RoadShoulder"},
    {"color": [0, 0, 192], "isthing": 1, "id": 20, "name": "Sidewalk"},
    {"color": [192, 128, 128], "isthing": 1, "id": 21, "name": "SignSymbol"},
    {"color": [128, 128, 128], "isthing": 1, "id": 22, "name": "Sky"},
    {"color": [64, 128,192], "isthing": 1, "id": 23, "name": "SUVPickupTruck"},
    {"color": [0, 0, 64], "isthing": 1, "id": 24, "name": "TrafficCone"},
    {"color": [0, 64, 64], "isthing": 1, "id": 25, "name": "TrafficLight"},
    {"color": [192, 64, 128], "isthing": 1, "id": 26, "name": "Train"},
    {"color": [128, 128, 0], "isthing": 1, "id": 27, "name": "Tree"},
    {"color": [192, 128, 192], "isthing": 1, "id": 28, "name": "Truck_Bus"},
    {"color": [64, 0, 64], "isthing": 1, "id": 29, "name": "Tunnel"},
    {"color": [192, 192, 0], "isthing": 1, "id": 30, "name": "VegetationMisc"},
    {"color": [0, 0, 0], "isthing": 1, "id": 31, "name": "Void"},
    {"color": [64, 192, 0], "isthing": 1, "id": 32, "name": "Wall"},]


def _get_camvid_coco_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing.
    stuff_ids = [k["id"] for k in CAMVID_CATEGORIES]
    assert len(stuff_ids) == 32, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in CAMVID_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret


def register_all_camvid_coco(root):
    root = os.path.join(root, "camvid")
    meta = _get_camvid_coco_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("train", "images/train", "labels/train"),
        ("test", "images/test", "labels/test"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"camvid_{name}_segmentation"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )


#_root = os.getenv("DETECTRON2_DATASETS", "/mnt/shared") #replaced 'datasets' with correct path
_root = os.getenv("DETECTRON2_DATASETS", "/mnt/raid1/dataset/") #replaced 'datasets' with correct path
register_all_camvid_coco(_root)
