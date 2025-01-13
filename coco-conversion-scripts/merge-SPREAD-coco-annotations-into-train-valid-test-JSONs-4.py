#!/usr/bin/env python3

import os
import json
import random
import shutil
from collections import defaultdict

def load_and_process_annotations(json_file):
    with open(json_file) as f:
        data = json.load(f)

    # Check if data is already a dictionary or just a list of annotations.
    if isinstance(data, list):  # Handle case where json_file contains only annotations
        # Create a dummy structure.  This assumes image_id and category_id exist
        # and are consistent across these "annotation-only" files, which might
        # not always be the case.  A more robust solution would be to extract
        # images and categories from a main file and reference them here.
        images = [{"id": ann["image_id"]} for ann in data]  # Unique image IDs
        categories = [{"id": ann["category_id"], "name": "unknown"} for ann in data]
         # Remove duplicate image and category IDs using dict.fromkeys
        images = list(dict.fromkeys(img["id"] for img in images))
        images = [{'id': img_id} for img_id in images] # Convert back into the proper format
        categories = list(dict.fromkeys(cat["id"] for cat in categories))
        categories = [{'id': cat_id, "name": "unknown"} for cat_id in categories] # Convert back

        data = {"images": images, "annotations": data, "categories": categories}

    return data


def merge_coco_annotations(base_dir):
    merged_data = {"images": [], "annotations": [], "categories": []}
    image_id_mapping = {}  # Map old image IDs to new, contiguous IDs
    annotation_id_mapping = {}  # Map old annotation IDs
    next_image_id = 0
    next_annotation_id = 0
    category_ids = set()

    # Traverse through the base directory
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                json_file_path = os.path.join(root, file)
                data = load_and_process_annotations(json_file_path)

                # Merge categories, keeping track of existing ones
                for category in data.get('categories', []): #.get for the case that categories is missing
                    if category['id'] not in category_ids:
                        merged_data['categories'].append(category)
                        category_ids.add(category['id'])


                # Merge images and remap image IDs
                for image in data.get('images', []):  #.get for case that images is missing
                    old_image_id = image['id']
                    if old_image_id not in image_id_mapping:
                        image_id_mapping[old_image_id] = next_image_id
                        new_image = image.copy() # Important to copy to avoid modifying the original
                        new_image['id'] = next_image_id
                        
                        # Try to get 'file_name' from the image, if present
                        if 'file_name' in new_image:
                            # Construct absolute path from relative path
                            new_image['file_name'] = os.path.join(root, new_image['file_name'])
                        else:
                            # Fallback if 'file_name' doesn't exist.  MAKE SURE THIS IS CORRECT for your data.
                            # This assumes the image file has the same basename as the JSON but with .jpg
                            new_image['file_name'] = os.path.join(root, os.path.splitext(file)[0] + ".jpg")
                            
                        merged_data['images'].append(new_image)
                        next_image_id += 1

                # Merge annotations and remap IDs
                for annotation in data.get('annotations', []):  #.get for case that annotations is missing
                    old_annotation_id = annotation['id']
                    annotation_id_mapping[old_annotation_id] = next_annotation_id
                    new_annotation = annotation.copy() # Use .copy() here too
                    new_annotation['id'] = next_annotation_id
                    new_annotation['image_id'] = image_id_mapping[annotation['image_id']]
                    merged_data['annotations'].append(new_annotation)
                    next_annotation_id += 1

    return merged_data


def split_dataset(data, train_ratio=0.7, valid_ratio=0.2):
    # Split image IDs instead of the entire annotation objects.
    image_ids = [image['id'] for image in data['images']]
    random.shuffle(image_ids)
    total = len(image_ids)
    train_end = int(train_ratio * total)
    valid_end = train_end + int(valid_ratio * total)

    train_image_ids = set(image_ids[:train_end])
    valid_image_ids = set(image_ids[train_end:valid_end])
    test_image_ids = set(image_ids[valid_end:])

    train_data = {"images": [], "annotations": [], "categories": data['categories']}
    valid_data = {"images": [], "annotations": [], "categories": data['categories']}
    test_data = {"images": [], "annotations": [], "categories": data['categories']}

    # Filter images and annotations based on the split image IDs
    for image in data['images']:
        if image['id'] in train_image_ids:
            train_data['images'].append(image)
        elif image['id'] in valid_image_ids:
            valid_data['images'].append(image)
        else:
            test_data['images'].append(image)

    for annotation in data['annotations']:
        if annotation['image_id'] in train_image_ids:
            train_data['annotations'].append(annotation)
        elif annotation['image_id'] in valid_image_ids:
            valid_data['annotations'].append(annotation)
        else:
            test_data['annotations'].append(annotation)

    return train_data, valid_data, test_data


def save_coco_annotations(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f)
        #json.dump(data, f, indent=4)  # Use indent=4 for readability if needed


def main(base_dir, output_dir):
    # Merge all annotations
    print("Merging annotations...")
    merged_data = merge_coco_annotations(base_dir)
    print(f"Total images: {len(merged_data['images'])}")
    print(f"Total annotations: {len(merged_data['annotations'])}")
    print(f"Total categories: {len(merged_data['categories'])}")

    # Split into train, validation, and test sets
    print("Splitting dataset...")
    train_data, valid_data, test_data = split_dataset(merged_data)
    print(f"Train images: {len(train_data['images'])}")
    print(f"Train annotations: {len(train_data['annotations'])}")
    print(f"Valid images: {len(valid_data['images'])}")
    print(f"Valid annotations: {len(valid_data['annotations'])}")
    print(f"Test images: {len(test_data['images'])}")
    print(f"Test annotations: {len(test_data['annotations'])}")
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the datasets to JSON files
    print("Saving datasets...")
    save_coco_annotations(train_data, os.path.join(output_dir, 'train_annotations.json'))
    save_coco_annotations(valid_data, os.path.join(output_dir, 'valid_annotations.json'))
    save_coco_annotations(test_data, os.path.join(output_dir, 'test_annotations.json'))

    print("Done!")


if __name__ == "__main__":
    # Adjust these paths as necessary
    base_directory   = '/mnt/raid1/dataset/spread/spread-v2'
    output_directory = '/mnt/raid1/dataset/spread/spread-v2'

    main(base_directory, output_directory)
