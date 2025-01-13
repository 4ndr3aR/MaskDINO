#!/usr/bin/env python3

import os
import json
from PIL import Image, ImageDraw
import numpy as np

# Define color mappings
GREEN_SHADE = (0, 255, 0)  # Base green color
BACKGROUND = (0, 0, 0)  # Black for background
TREE = (0, 255, 0)  # Green for tree class

def generate_instance_mask(image_width, image_height, annotations, output_path):
    """
    Generate an instance segmentation mask from COCO annotations.
    Each instance is assigned a different shade of green.
    """
    mask = Image.new("RGB", (image_width, image_height), color=BACKGROUND)
    draw = ImageDraw.Draw(mask)

    for idx, ann in enumerate(annotations):
        # Generate a unique shade of green for each instance
        shade = (0, 255 - (idx * 10), 0)  # Adjust green intensity
        if shade[1] < 0:  # Ensure shade stays within valid range
            shade = (0, 0, 0)  # Fallback to black if too many instances

        # Draw segmentation mask for the instance
        for seg in ann["segmentation"]:
            points = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
            draw.polygon(points, fill=shade, outline=shade)

        # Save bounding box in a text file
        bbox = ann["bbox"]  # [x, y, width, height]
        x1, y1, w, h = map(int, bbox)
        x2, y2 = x1 + w, y1 + h
        bbox_str = f"{shade[0]} {shade[1]} {shade[2]} {x1} {y1} {x2} {y2}\n"
        with open(os.path.join(output_path, f"{os.path.splitext(os.path.basename(output_path))[0]}_bbox.txt"), "a") as f:
            f.write(bbox_str)

    mask.save(os.path.join(output_path, f"{os.path.splitext(os.path.basename(output_path))[0]}_instance_mask.png"))

def generate_semantic_mask(image_width, image_height, annotations, output_path):
    """
    Generate a semantic segmentation mask from COCO annotations.
    Use black for background and green for tree class.
    """
    mask = Image.new("RGB", (image_width, image_height), color=BACKGROUND)
    draw = ImageDraw.Draw(mask)

    for ann in annotations:
        for seg in ann["segmentation"]:
            points = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
            draw.polygon(points, fill=TREE, outline=TREE)

    mask.save(os.path.join(output_path, f"{os.path.splitext(os.path.basename(output_path))[0]}_semantic_mask.png"))

def process_dataset(root_dir):
    """
    Process the dataset directory and generate instance and semantic segmentation masks.
    """
    rgb_dir = os.path.join(root_dir, "rgb")
    coco_dir = os.path.join(root_dir, "coco_annotation")
    output_dir = os.path.join(root_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all JSON files in the COCO annotation directory
    for json_file in os.listdir(coco_dir):
        if json_file.endswith(".json"):
            json_path = os.path.join(coco_dir, json_file)
            with open(json_path, "r") as f:
                coco_data = json.load(f)

            # Extract image metadata
            image_info = coco_data["images"][0]  # Assuming one image per JSON file
            image_width = image_info["width"]
            image_height = image_info["height"]
            image_name = image_info["file_name"]

            # Extract annotations
            annotations = coco_data["annotations"]

            # Generate instance segmentation mask and bounding boxes
            instance_output_path = os.path.join(output_dir, os.path.splitext(image_name)[0])
            os.makedirs(instance_output_path, exist_ok=True)
            generate_instance_mask(image_width, image_height, annotations, instance_output_path)

            # Generate semantic segmentation mask
            generate_semantic_mask(image_width, image_height, annotations, instance_output_path)

    print("Processing complete. Outputs saved in the 'output' directory.")

# Main execution
if __name__ == "__main__":
    root_dir = "/mnt/raid1/dataset/spread/zipfiles/v2/birch_forest"  # Replace with your dataset directory path
    process_dataset(root_dir)
