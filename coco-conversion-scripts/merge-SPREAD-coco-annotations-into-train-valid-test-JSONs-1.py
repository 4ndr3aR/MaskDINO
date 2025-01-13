#!/usr/bin/env python3


import os
import json
import random

# Define the root directory of your dataset
root_dir = "/mnt/raid1/dataset/spread/zipfiles/v2/spread-v2"

# Define the output directory for the merged JSON files
output_dir = '/mnt/raid1/dataset/spread/zipfiles/v2/spread-v2'

# Define the split ratios
train_ratio = 0.7
val_ratio   = 0.2
test_ratio  = 0.1

# Initialize lists to store the annotations
dataset_annotations = []
train_annotations   = []
val_annotations     = []
test_annotations    = []

# Walk through the directory tree
for root, dirs, files in os.walk(root_dir):
    #print(f'{root}, {dirs}, {files}')
    # Check if the current directory contains JSON files
    if 'coco_annotation' in dirs:
        annotations_dir = os.path.join(root, 'coco_annotation')
        for idx, file in enumerate(os.listdir(annotations_dir)):
            if file.endswith('.json'):
                # Load the JSON file
                with open(os.path.join(annotations_dir, file), 'r') as f:
                    annotations = json.load(f)
                    # Add the annotations to the list
                    dataset_annotations.append(annotations)
        print(f'Found {idx + 1} JSON files in {annotations_dir}.')

print(f'Found {len(dataset_annotations)} annotations in the dataset.')

# Shuffle the annotations
random.shuffle(dataset_annotations)

# Split the annotations into train, val, and test sets
train_size		= int(len(dataset_annotations) * train_ratio)
val_size		= int(len(dataset_annotations) * val_ratio)
train_annotations	= dataset_annotations[:train_size]
val_annotations		= dataset_annotations[train_size:train_size + val_size]
test_annotations	= dataset_annotations[train_size + val_size:]

print(f"Training set  : {len(train_annotations)} annotations")
print(f"Validation set: {len(val_annotations)} annotations")
print(f"Test set      : {len(test_annotations)} annotations")

# Save the annotations to JSON files
with open(os.path.join(output_dir, 'train.json'), 'w') as f:
    json.dump(train_annotations, f)

with open(os.path.join(output_dir, 'val.json'), 'w') as f:
    json.dump(val_annotations, f)

with open(os.path.join(output_dir, 'test.json'), 'w') as f:
    json.dump(test_annotations, f)

print("Annotations merged and split into train, val, and test sets.")

