#!/usr/bin/env python3

import os
import json
import random
import shutil
from collections import defaultdict

def load_annotations_from_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def merge_and_unify_ids(base_dir):
    """
    Traverse base_dir, read all COCO annotation files, and unify them into one COCO dataset
    with new IDs for images, annotations, and categories so that there are no duplicates.
    """

    # Temporary storage to read all data first
    all_coco = []
    
    # 1. Collect all COCO data from each JSON file
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                json_file_path = os.path.join(root, file)
                coco_data = load_annotations_from_json(json_file_path)
                
                # Validate minimal COCO fields
                if not all(k in coco_data for k in ["images", "annotations", "categories"]):
                    raise ValueError(f"Invalid COCO format in {json_file_path}")
                
                all_coco.append(coco_data)
    
    # 2. We will create new IDs for:
    #    - categories
    #    - images
    #    - annotations
    #
    #    We'll unify categories by their names (assuming the same category name means the same category).
    #    Then we'll reassign IDs for images and annotations.

    # 2a. Unify categories by name
    category_name2new_id = {}
    new_categories       = []
    next_cat_id          = 1
    
    # We'll gather all category definitions.  If a name is repeated, we assume it's the same category.
    for coco_data in all_coco:
        for cat in coco_data["categories"]:
            cat_name = cat["name"]
            if cat_name not in category_name2new_id:
                category_name2new_id[cat_name] = next_cat_id
                new_categories.append({
                    "id"           : next_cat_id,
                    "name"         : cat_name,
                    "supercategory": cat.get("supercategory", "")
                })
                next_cat_id += 1

    # 2b. Unify images and annotations
    new_images       = []
    new_annotations  = []
    next_image_id    = 1
    next_annot_id    = 1

    # To avoid re-creating the same image multiple times if it appears in multiple JSON,
    # we can deduplicate by (file_name, height, width). If you have collisions
    # (same file_name but different actual images), you may need a different key strategy.
    image_key2new_id = {}  # (file_name, height, width) -> new image_id

    for coco_data in all_coco:
        # Build a local map for the original cat_id => new cat_id (via cat name).
        # This helps convert the annotation "category_id" to the newly assigned ID.
        oldcat2newcat = {}
        for cat in coco_data["categories"]:
            oldcat_id             = cat["id"]
            cat_name              = cat["name"]
            oldcat2newcat[oldcat_id] = category_name2new_id[cat_name]

        # Build a local map of old_image_id => new_image_id for this JSON file
        oldimg2newimg = {}

        # Process images
        for img in coco_data["images"]:
            f_name  = img["file_name"]
            width   = img["width"]
            height  = img["height"]
            key     = (f_name, width, height)

            if key not in image_key2new_id:
                image_key2new_id[key] = next_image_id
                new_images.append({
                    "id"        : next_image_id,
                    "file_name" : f_name,
                    "height"    : height,
                    "width"     : width
                })
                next_image_id += 1
            
            # Map the old image id to the new image id
            oldimg2newimg[img["id"]] = image_key2new_id[key]

        # Process annotations
        for ann in coco_data["annotations"]:
            old_img_id    = ann["image_id"]
            old_cat_id    = ann["category_id"]
            new_img_id    = oldimg2newimg[old_img_id]
            new_cat_id    = oldcat2newcat[old_cat_id]

            # Create a brand new annotation
            new_ann = {
                "id"         : next_annot_id,
                "image_id"   : new_img_id,
                "category_id": new_cat_id,
                "bbox"       : ann["bbox"],
                "area"       : ann.get("area", 0),
                "iscrowd"    : ann.get("iscrowd", 0)
            }
            # If you'd like to preserve other annotation fields (e.g., segmentation, keypoints),
            # copy them here as well:
            #
            # if "segmentation" in ann:
            #     new_ann["segmentation"] = ann["segmentation"]
            # etc.

            new_annotations.append(new_ann)
            next_annot_id += 1

    # 3. Return the unified COCO dataset
    merged_coco_data = {
        "images"     : new_images,
        "annotations": new_annotations,
        "categories" : new_categories
    }

    return merged_coco_data

def split_dataset(coco_data, train_ratio=0.7, valid_ratio=0.2, seed=42):
    """Splits the dataset into train/val/test by shuffling the images."""
    random.seed(seed)  # For reproducible splits
    random.shuffle(coco_data['images'])

    total_images = len(coco_data['images'])
    train_end    = int(train_ratio * total_images)
    valid_end    = train_end + int(valid_ratio * total_images)

    train_images = coco_data['images'][:train_end]
    valid_images = coco_data['images'][train_end:valid_end]
    test_images  = coco_data['images'][valid_end:]

    train_ids = set(img['id'] for img in train_images)
    valid_ids = set(img['id'] for img in valid_images)
    test_ids  = set(img['id'] for img in test_images)

    # Filter annotations
    train_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in train_ids]
    valid_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in valid_ids]
    test_annotations  = [ann for ann in coco_data['annotations'] if ann['image_id'] in test_ids]

    train_set = {
        "images"     : train_images,
        "annotations": train_annotations,
        "categories" : coco_data['categories']
    }
    valid_set = {
        "images"     : valid_images,
        "annotations": valid_annotations,
        "categories" : coco_data['categories']
    }
    test_set = {
        "images"     : test_images,
        "annotations": test_annotations,
        "categories" : coco_data['categories']
    }

    return train_set, valid_set, test_set

def save_coco_annotations(coco_data, output_file):
    """Helper to save a COCO dictionary as JSON."""
    with open(output_file, 'w') as f:
        json.dump(coco_data, f)

def main(base_dir, output_dir):
    # Step 1: Merge all annotations while reassigning IDs to avoid duplicates
    print("Merging & Unifying annotations...")
    merged_coco_data = merge_and_unify_ids(base_dir)
    print(f"Total number of images     : {len(merged_coco_data['images'])}")
    print(f"Total number of annotations: {len(merged_coco_data['annotations'])}")
    print(f"Total number of categories : {len(merged_coco_data['categories'])}")

    # Step 2: Split into train, validation, and test
    print("Splitting dataset...")
    train_set, valid_set, test_set = split_dataset(merged_coco_data)
    print(f"Train images: {len(train_set['images'])}")
    print(f"Valid images: {len(valid_set['images'])}")
    print(f"Test  images: {len(test_set['images'])}")

    # Step 3: Save the split datasets
    os.makedirs(output_dir, exist_ok=True)

    print("Saving datasets...")
    save_coco_annotations(train_set, os.path.join(output_dir, 'train_annotations-7.json'))
    save_coco_annotations(valid_set, os.path.join(output_dir, 'valid_annotations-7.json'))
    save_coco_annotations(test_set , os.path.join(output_dir, 'test_annotations-7.json'))

    print("Done!")

if __name__ == "__main__":
    base_directory   = '/mnt/raid1/dataset/spread/spread-v2'  # Adjust as needed
    output_directory = '/mnt/raid1/dataset/spread/spread-v2'  # Adjust as needed

    main(base_directory, output_directory)
