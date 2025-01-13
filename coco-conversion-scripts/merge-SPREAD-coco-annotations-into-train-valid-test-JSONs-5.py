#!/usr/bin/env python3

import os
import json
import random
import shutil
from collections import defaultdict


def load_annotations_and_images(base_dir):
    """Loads annotations and image info, handling potential multiple JSON files per directory."""
    all_annotations = []
    all_images = []
    all_categories = []
    category_name_to_id = {}  # Keep track of category names and their IDs
    next_category_id = 1      # Start category IDs from 1
    next_image_id = 1  # start image ids from 1
    next_annotation_id = 1

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                json_file_path = os.path.join(root, file)
                with open(json_file_path, 'r') as f:
                    data = json.load(f)

                    # Handle potential list of annotations within a single file
                    if isinstance(data, list):  # If it's a list of annotations
                        for annotation in data:
                            # Extract category name and ID
                            category_name = annotation.get("category_name")
                            if not category_name:
                                print(f"Warning: Skipping annotation without category_name in {json_file_path}")
                                continue  # Skip annotations without a category name
                            
                            category_id = category_name_to_id.get(category_name)
                            if category_id is None: # new category
                                category_id = next_category_id
                                category_name_to_id[category_name] = category_id
                                category = {
                                    "id": category_id,
                                    "name": category_name,
                                    "supercategory": "object"  # Or a suitable supercategory
                                }
                                all_categories.append(category)
                                next_category_id += 1

                            # Process the image data
                            image_path = os.path.join(root, annotation['file_name'])  # Full path
                            image_id = annotation.get('image_id') #check if image_id is already there
                            if image_id is None: #id not present, set it
                                image_id = next_image_id
                                annotation['image_id'] = image_id
                                next_image_id+=1

                            image_data = {
                                "id": image_id,
                                "file_name": image_path,  # Store the full path relative to base_dir
                                "width": annotation.get('width', 0),   # Get width if present, else default
                                "height": annotation.get('height', 0)  # Get height if present, else default
                            }
                            if any(img["id"] == image_id for img in all_images):
                                pass
                                # print(f'image with id {image_id} already present')
                            else:
                                all_images.append(image_data)

                            # Process the annotation data, and add the category ID
                            annotation_entry = {
                                "id": next_annotation_id,
                                "image_id": image_id,
                                "category_id": category_id,  # Use the assigned category ID
                                "bbox": annotation['bbox'],
                                "area": annotation['bbox'][2] * annotation['bbox'][3],  # Calculate area
                                "iscrowd": annotation.get('iscrowd', 0),  # Default to 0 if not present
                                # Add other annotation fields as needed
                            }
                            all_annotations.append(annotation_entry)
                            next_annotation_id+=1
                    else:  # Handle other potential JSON structures if needed
                         print(f"Warning: Unexpected JSON structure in {json_file_path}")
                         # Add error handling or logging as appropriate


    return all_images, all_annotations, all_categories


def split_dataset(images, annotations, train_ratio=0.7, valid_ratio=0.2):
    """Splits the dataset into train, validation, and test sets based on image IDs."""

    # Group annotations by image ID for efficient splitting
    annotations_by_image_id = defaultdict(list)
    for ann in annotations:
        annotations_by_image_id[ann['image_id']].append(ann)

    random.shuffle(images)  # Shuffle the images
    total_images = len(images)
    train_end = int(train_ratio * total_images)
    valid_end = train_end + int(valid_ratio * total_images)

    train_images = images[:train_end]
    valid_images = images[train_end:valid_end]
    test_images = images[valid_end:]

    # Collect annotations corresponding to the split image sets
    train_annotations = []
    for img in train_images:
        train_annotations.extend(annotations_by_image_id[img['id']])
    valid_annotations = []
    for img in valid_images:
        valid_annotations.extend(annotations_by_image_id[img['id']])
    test_annotations = []
    for img in test_images:
        test_annotations.extend(annotations_by_image_id[img['id']])


    return (train_images, train_annotations), (valid_images, valid_annotations), (test_images, test_annotations)

def create_coco_dict(images, annotations, categories):
    """Creates a COCO-formatted dictionary."""
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        # You might optionally include other COCO fields like "info", "licenses", etc.
    }

def save_coco_json(coco_dict, output_file):
    """Saves the COCO dictionary to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(coco_dict, f, indent=4)

def main(base_dir, output_dir):
    print("Loading annotations and images...")
    images, annotations, categories = load_annotations_and_images(base_dir)
    print(f'Total number of images: {len(images)}')
    print(f'Total number of annotations: {len(annotations)}')
    print(f'Total number of categories: {len(categories)}')
    
    print("Splitting dataset...")
    (train_images, train_annotations), (valid_images, valid_annotations), (test_images, test_annotations) = split_dataset(images, annotations)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create and save COCO datasets
    print("Saving datasets...")
    train_coco = create_coco_dict(train_images, train_annotations, categories)
    save_coco_json(train_coco, os.path.join(output_dir, 'train_annotations-5.json'))

    valid_coco = create_coco_dict(valid_images, valid_annotations, categories)
    save_coco_json(valid_coco, os.path.join(output_dir, 'valid_annotations-5.json'))

    test_coco = create_coco_dict(test_images, test_annotations, categories)
    save_coco_json(test_coco, os.path.join(output_dir, 'test_annotations-5.json'))

    print("Done!")

if __name__ == "__main__":
    # Adjust these paths as necessary
    base_directory   = '/mnt/raid1/dataset/spread/spread-v2'
    output_directory = '/mnt/raid1/dataset/spread/spread-v2'

    main(base_directory, output_directory)
