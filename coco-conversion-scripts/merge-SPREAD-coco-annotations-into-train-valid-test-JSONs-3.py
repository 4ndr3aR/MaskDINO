#!/usr/bin/env python3

import os
import json
import random
import shutil
from collections import defaultdict

def load_annotations_from_json(json_file):
    with open(json_file) as f:
        return json.load(f)

def merge_annotations_coco(base_dir):
    merged_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    category_mapping = {}
    next_category_id = 1
    next_image_id = 1
    next_annotation_id = 1

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                json_file_path = os.path.join(root, file)
                data = load_annotations_from_json(json_file_path)
                #print(f'{json_file_path}')
                #print(f'{type(data)}')
                #print(f'{data}')

                # Process categories
                for category in data.get("categories", []):
                    if category['name'] not in category_mapping:
                        category_mapping[category['name']] = next_category_id
                        category['id'] = next_category_id
                        merged_data['categories'].append(category)
                        next_category_id += 1
                    else:
                        category['id'] = category_mapping[category['name']] # Keep the existing ID from mapping

                # Process images and annotations, updating IDs and linking
                for image in data.get("images", []):
                    image['id'] = next_image_id
                    merged_data['images'].append(image)
                    
                    # Process annotations related to the current image
                    for annotation in data.get("annotations", []):
                        if annotation['image_id'] == image['id']: # Assuming original IDs were consistent within each file
                            annotation['id'] = next_annotation_id
                            annotation['image_id'] = next_image_id # Update to new image ID
                            annotation['category_id'] = category_mapping[next((cat['name'] for cat in data.get("categories", []) if cat['id'] == annotation['category_id']), None)] # Update to consistent category ID
                            merged_data['annotations'].append(annotation)
                            next_annotation_id += 1
                    next_image_id += 1
                    
    return merged_data

def split_dataset_coco(data, train_ratio=0.7, valid_ratio=0.2):
    # Shuffle image IDs
    image_ids = [img['id'] for img in data['images']]
    random.shuffle(image_ids)
    
    total     = len(image_ids)
    train_end = int(train_ratio * total)
    valid_end = train_end + int(valid_ratio * total)
    
    train_image_ids = set(image_ids[:train_end])
    valid_image_ids = set(image_ids[train_end:valid_end])
    test_image_ids  = set(image_ids[valid_end:])
    
    train_data = {"images": [], "annotations": [], "categories": data['categories']}
    valid_data = {"images": [], "annotations": [], "categories": data['categories']}
    test_data  = {"images": [], "annotations": [], "categories": data['categories']}

    # Filter images and annotations based on shuffled IDs
    for image in data['images']:
        if image['id'] in train_image_ids:
            train_data['images'].append(image)
        elif image['id'] in valid_image_ids:
            valid_data['images'].append(image)
        elif image['id'] in test_image_ids:
            test_data['images'].append(image)
            
    for annotation in data['annotations']:
        if annotation['image_id'] in train_image_ids:
            train_data['annotations'].append(annotation)
        elif annotation['image_id'] in valid_image_ids:
            valid_data['annotations'].append(annotation)
        elif annotation['image_id'] in test_image_ids:
            test_data['annotations'].append(annotation)
    
    return train_data, valid_data, test_data

def save_coco_annotations(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

def main(base_dir, output_dir):
    # Merge all annotations into COCO format
    print("Merging annotations into COCO format...")
    merged_data = merge_annotations_coco(base_dir)
    print(f'Total number of images		: {len(merged_data["images"])}')
    print(f'Total number of annotations		: {len(merged_data["annotations"])}')
    print(f'Total number of categories		: {len(merged_data["categories"])}')

    # Split into train, validation, and test sets
    print("Splitting dataset...")
    train_data, valid_data, test_data = split_dataset_coco(merged_data)
    print(f'Number of images in train set	: {len(train_data["images"])}')
    print(f'Number of annotations in train set	: {len(train_data["annotations"])}')
    print(f'Number of images in valid set	: {len(valid_data["images"])}')
    print(f'Number of annotations in valid set	: {len(valid_data["annotations"])}')
    print(f'Number of images in test set	: {len(test_data["images"])}')
    print(f'Number of annotations in test set	: {len(test_data["annotations"])}')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the datasets to COCO format JSON files
    print("Saving datasets...")
    save_coco_annotations(train_data, os.path.join(output_dir, 'train_annotations.json'))
    save_coco_annotations(valid_data, os.path.join(output_dir, 'valid_annotations.json'))
    save_coco_annotations(test_data,  os.path.join(output_dir, 'test_annotations.json'))

    print("Done!")

if __name__ == "__main__":
    base_directory   = '/mnt/raid1/dataset/spread/spread-v2'
    output_directory = '/mnt/raid1/dataset/spread/spread-v2'

    main(base_directory, output_directory)
