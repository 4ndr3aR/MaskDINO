#!/usr/bin/env python3

import os
import json
import random
from collections import defaultdict

def load_annotations_from_json(json_file):
    with open(json_file) as f:
        return json.load(f)

def merge_annotations(base_dir):
    images = []
    annotations = []
    categories = []

    # Traverse through the base directory
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                json_file_path = os.path.join(root, file)
                coco_data = load_annotations_from_json(json_file_path)
                
                # Ensure the loaded file follows COCO format
                if 'images' not in coco_data or 'annotations' not in coco_data or 'categories' not in coco_data:
                    raise ValueError(f"Invalid COCO format in {json_file_path}")

                images.extend(coco_data['images'])
                annotations.extend(coco_data['annotations'])
                categories.extend(coco_data['categories'])

    # Deduplicate categories (assuming they are the same across all files)
    unique_categories = list({category['id']: category for category in categories}.values())

    return {
        'images'        : images,
        'annotations'   : annotations,
        'categories'    : unique_categories
    }

def split_dataset(coco_data, train_ratio=0.7, valid_ratio=0.2):
    random.shuffle(coco_data['images'])  # Shuffle the images

    total_images = len(coco_data['images'])
    train_end    = int(train_ratio * total_images)
    valid_end    = train_end + int(valid_ratio * total_images)

    train_images = coco_data['images'][:train_end]
    valid_images = coco_data['images'][train_end:valid_end]
    test_images  = coco_data['images'][valid_end:]

    # Create image ID sets for fast lookup
    train_image_ids = set(image['id'] for image in train_images)
    valid_image_ids = set(image['id'] for image in valid_images)
    test_image_ids  = set(image['id'] for image in test_images)

    # Split annotations based on image IDs
    train_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in train_image_ids]
    valid_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in valid_image_ids]
    test_annotations  = [ann for ann in coco_data['annotations'] if ann['image_id'] in test_image_ids]

    # Reassign unique IDs to the annotations
    annotation_id = 1
    for ann in train_annotations:
        ann['id'] = annotation_id
        annotation_id += 1
    for ann in valid_annotations:
        ann['id'] = annotation_id
        annotation_id += 1
    for ann in test_annotations:
        ann['id'] = annotation_id
        annotation_id += 1

    # Return the split datasets with the original categories
    return (
        {'images': train_images, 'annotations': train_annotations, 'categories': coco_data['categories']},
        {'images': valid_images, 'annotations': valid_annotations, 'categories': coco_data['categories']},
        {'images': test_images , 'annotations': test_annotations , 'categories': coco_data['categories']}
    )

def save_coco_annotations(coco_data, output_file):
    with open(output_file, 'w') as f:
        json.dump(coco_data, f)

def main(base_dir, output_dir):
    # Merge all annotations into a single COCO-format dictionary
    print("Merging annotations...")
    merged_coco_data = merge_annotations(base_dir)
    print(f'Total number of images     : {len(merged_coco_data["images"])}')
    print(f'Total number of annotations: {len(merged_coco_data["annotations"])}')
    print(f'Total number of categories : {len(merged_coco_data["categories"])}')

    # Split into train, validation, and test sets
    print("Splitting dataset...")
    train_set, valid_set, test_set = split_dataset(merged_coco_data)
    print(f'Total number of images in train set: {len(train_set["images"])}')
    print(f'Total number of images in valid set: {len(valid_set["images"])}')
    print(f'Total number of images in test set : {len(test_set["images"])}')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the datasets to JSON files
    print("Saving datasets...")
    save_coco_annotations(train_set, os.path.join(output_dir, 'train_annotations-10.json'))
    save_coco_annotations(valid_set, os.path.join(output_dir, 'valid_annotations-10.json'))
    save_coco_annotations(test_set , os.path.join(output_dir, 'test_annotations-10.json'))

    print("Done!")

if __name__ == "__main__":
    # Adjust these paths as necessary
    base_directory   = '/mnt/raid1/dataset/spread/spread-v2'
    output_directory = '/mnt/raid1/dataset/spread/spread-v2'

    main(base_directory, output_directory)
