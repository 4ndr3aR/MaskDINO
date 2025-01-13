#!/usr/bin/env python3

import os
import json
import random
import shutil

def load_annotations_from_json(json_file):
    with open(json_file) as f:
        return json.load(f)

def merge_annotations(base_dir):
    annotations = []
    # Traverse through the base directory
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                json_file_path = os.path.join(root, file)
                annotations.append(load_annotations_from_json(json_file_path))
    return annotations

def split_dataset(annotations, train_ratio=0.7, valid_ratio=0.2):
    random.shuffle(annotations)  # Shuffle the dataset
    total = len(annotations)
    train_end = int(train_ratio * total)
    valid_end = train_end + int(valid_ratio * total)
    
    train_set = annotations[:train_end]
    valid_set = annotations[train_end:valid_end]
    test_set = annotations[valid_end:]
    
    return train_set, valid_set, test_set

def save_annotations(annotations, output_file):
    with open(output_file, 'w') as f:
        json.dump(annotations, f)
        #json.dump(annotations, f, indent=4)

def main(base_dir, output_dir):
    # Merge all annotations
    print("Merging annotations...")
    merged_annotations = merge_annotations(base_dir)
    print(f'Total number of annotations: {len(merged_annotations)}')
    
    # Split into train, validation, and test sets
    print("Splitting dataset...")
    train_set, valid_set, test_set = split_dataset(merged_annotations)
    print(f'Total number of annotations in train set: {len(train_set)}')
    print(f'Total number of annotations in valid set: {len(valid_set)}')
    print(f'Total number of annotations in test set: {len(test_set)}')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the datasets to JSON files
    print("Saving datasets...")
    save_annotations(train_set, os.path.join(output_dir, 'train_annotations.json'))
    save_annotations(valid_set, os.path.join(output_dir, 'valid_annotations.json'))
    save_annotations(test_set,  os.path.join(output_dir, 'test_annotations.json'))
    
    print("Done!")

if __name__ == "__main__":
    # Adjust these paths as necessary
    base_directory   = '/mnt/raid1/dataset/spread/zipfiles/v2/spread-v2'
    output_directory = '/mnt/raid1/dataset/spread/zipfiles/v2/spread-v2'

    main(base_directory, output_directory)
