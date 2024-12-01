#
#!/usr/bin/env python
#-*- coding: utf-8 -*-
#
# ==============================================================================
# Copyright (c) 2024 Nextera Robotic Systems
# Description: This script copies the annotations and images to a single folder and json file
#
# Author: Andrew Kent
# Created on: Thu Mar 07 2024 5:47:50 PM
# ==============================================================================
#

import matplotlib.pyplot as plt

from tqdm import tqdm

import argparse
import random
import shutil
import json
import glob
import os

def main(safety_set, split_ratio, images_annotation_split):

    # Make the directory for saving the jsons and plots
    os.makedirs(f'{safety_set}', exist_ok=True)

    # Gather all the annotation jsons in the data folder
    annotations = gather_annotations(safety_set)

    # Split the annotations into train and val
    if images_annotation_split == 'images':
        train, val, train_counts, val_counts, categories = split_images(safety_set, annotations, split_ratio)
    elif images_annotation_split == 'annotations':
        train, val, train_counts, val_counts, categories = split_annotations(safety_set, annotations, split_ratio)
    else: 
        return

    # Plot the category distribution
    plot_category_distribution(safety_set, train_counts, val_counts, categories)

    # Move the images to the train and val folders
    move_images(train, f'{safety_set}_train')
    move_images(val, f'{safety_set}_val')

    # Check to make sure all the annotations' images are in the images folder
    check_images_annotations(train, f'{safety_set}_train')
    check_images_annotations(val, f'{safety_set}_val')

def check_images_annotations(data_set, folder):

    # Get a map from image_id to image_filename
    image_id_to_filename = {image['id']: image['file_name'] for image in data_set['images']}

    # Check to see if the image exists in the image list, and if the image exists in the folder
    for annotation in tqdm(data_set['annotations'], desc=f"Checking Annotations in {folder} Folder"):
        image_id = annotation['image_id']
        if image_id not in image_id_to_filename:
            print(f"Image ID {image_id} not in {folder} set")
        else:
            image_filename = image_id_to_filename[image_id]
            if not os.path.isfile(f'data/{folder}_images/{image_filename}'):
                print(f"Image {image_filename} not in {folder} folder")

def calculate_bbox_area(bbox):
    return bbox[2] * bbox[3]

def split_images(safety_set, annotations, split_ratio):
    random.seed(42)
    
    # Get the list of category names
    categories = [category['name'] for category in annotations['categories']]

    # Initialize sets
    train = {'images': [], 'annotations': [], 'categories': annotations['categories']}
    val = {'images': [], 'annotations': [], 'categories': annotations['categories']}
    train_counts = {category['id']: 0 for category in annotations['categories']}
    val_counts = {category['id']: 0 for category in annotations['categories']}
    
    # Mapping of image IDs to images and annotations
    image_dict = {image['id']: image for image in annotations['images']}
    image_annotations = {image['id']: [] for image in annotations['images']}
    for annotation in annotations['annotations']:
        image_annotations[annotation['image_id']].append(annotation)
    
    # Calculate pixel counts for each image by category
    image_pixels = {}
    for image_id, anns in tqdm(image_annotations.items(), desc="Calculating Image Pixels"):
        image_pixels[image_id] = {cat['id']: 0 for cat in annotations['categories']}
        for ann in anns:
            cat_id = ann['category_id']
            image_pixels[image_id][cat_id] += calculate_bbox_area(ann['bbox'])

    # Shuffle and split images ensuring balanced category pixel counts
    all_images = list(image_dict.values())
    random.shuffle(all_images)
    total_pixels = {category['id']: 0 for category in annotations['categories']}
    for image in all_images:
        image_id = image['id']
        for cat_id in image_pixels[image_id]:
            total_pixels[cat_id] += image_pixels[image_id][cat_id]

    target_pixels = {cat_id: total * split_ratio for cat_id, total in total_pixels.items()}

    # Assign images to train or validation set
    image_added = set()
    for image in tqdm(all_images, desc="Splitting Images"):
        image_id = image['id']
        if image_id in image_added:
            continue  # Skip if already added to a set

        feasible_for_train = True
        for cat_id in image_pixels[image_id]:
            if train_counts[cat_id] + image_pixels[image_id][cat_id] > target_pixels[cat_id]:
                feasible_for_train = False
                break

        if feasible_for_train:
            train['images'].append(image)
            train['annotations'].extend(image_annotations[image_id])
            image_added.add(image_id)
            for cat_id in image_pixels[image_id]:
                train_counts[cat_id] += image_pixels[image_id][cat_id]
        else:
            val['images'].append(image)
            val['annotations'].extend(image_annotations[image_id])
            image_added.add(image_id)
            for cat_id in image_pixels[image_id]:
                val_counts[cat_id] += image_pixels[image_id][cat_id]

    # Save to JSON files
    with open(f'data/annotations/{safety_set}_train.json', 'w') as f:
        json.dump(train, f, indent=4, sort_keys=True)
    with open(f'data/annotations/{safety_set}_val.json', 'w') as f: 
        json.dump(val, f, indent=4, sort_keys=True)

    with open(f'{safety_set}/{safety_set}_train.json', 'w') as f:
        json.dump(train, f, indent=4, sort_keys=True)
    with open(f'{safety_set}/{safety_set}_val.json', 'w') as f: 
        json.dump(val, f, indent=4, sort_keys=True)

    # Assertion to check unique filenames
    train_filenames = set([image['file_name'] for image in train['images']])
    val_filenames = set([image['file_name'] for image in val['images']])
    assert train_filenames.isdisjoint(val_filenames), "Train and Val sets have overlapping image filenames"

    # Specify the number of validation and training images
    val_images = len(val['images'])
    train_images = len(train['images'])
    image_ratio = train_images / (train_images+val_images)

    # Print the number of validation and training images, and the ratio next to the split ratio
    print(f"Train: {train_images} images, Val {val_images} Images, Image Ratio {image_ratio}, Ratio {split_ratio}")

    return train, val, train_counts, val_counts, categories

def split_annotations(safety_set, annotations, split_ratio):

    # Set the random seed for reproducibility
    random.seed(42)  

    # Get the list of category names
    categories = [category['name'] for category in annotations['categories']]

    # Initialize the train and val sets
    train = {'images': [], 'annotations': [], 'categories': annotations['categories']}
    val = {'images': [], 'annotations': [], 'categories': annotations['categories']}
    train_counts = {category['id']: 0 for category in annotations['categories']}
    val_counts = {category['id']: 0 for category in annotations['categories']}
    
    # Create a mapping of image IDs to images
    image_dict = {image['id']: image for image in annotations['images']}

    # Group images and annotations by category
    category_to_images = {}
    category_to_annotations = {}
    for annotation in tqdm(annotations['annotations'], desc="Grouping Annotations"):
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        if category_id not in category_to_images:
            category_to_images[category_id] = []
            category_to_annotations[category_id] = []
        # Remoove leading folders from image file name
        image_dict[image_id]['file_name'] = image_dict[image_id]['file_name'].split('/')[-1]
        category_to_images[category_id].append(image_dict[image_id])
        category_to_annotations[category_id].append(annotation)
    
    for category_id in tqdm(category_to_images.keys(), desc="Processing Categories"):

        # Shuffle images and annotations together
        combined = list(zip(category_to_images[category_id], category_to_annotations[category_id]))
        random.shuffle(combined)
        
        # Unzip the shuffled tuples back into images and annotations
        shuffled_images, shuffled_annotations = zip(*combined)
        
        # Calculate split index
        split_index = int(len(shuffled_images) * split_ratio)
        
        # Append to train and val sets
        train['images'].extend(shuffled_images[:split_index])
        train['annotations'].extend(shuffled_annotations[:split_index])
        val['images'].extend(shuffled_images[split_index:])
        val['annotations'].extend(shuffled_annotations[split_index:])
        
        # Count categories
        train_counts[category_id] += len(shuffled_annotations[:split_index])
        val_counts[category_id] += len(shuffled_annotations[split_index:])

    # Save the train and val annotations to json files
    with open(f'data/annotations/{safety_set}_train.json', 'w') as f:
        json.dump(train, f, indent=4, sort_keys=True)
    with open(f'data/annotations/{safety_set}_val.json', 'w') as f: 
        json.dump(val, f, indent=4, sort_keys=True)

    with open(f'{safety_set}/{safety_set}_train.json', 'w') as f:
        json.dump(train, f, indent=4, sort_keys=True)
    with open(f'{safety_set}/{safety_set}_val.json', 'w') as f: 
        json.dump(val, f, indent=4, sort_keys=True)

    return train, val, train_counts, val_counts, categories

def plot_category_distribution(safety_set, train_counts, val_counts, categories_names):

    # Get the list of categories
    categories = list(train_counts.keys())
    train_vals = [train_counts[cat] for cat in categories]
    val_vals = [val_counts[cat] for cat in categories]

    # Get the total number of bboxes in train and val
    total_train_bboxes = int(sum(train_vals))
    total_val_bboxes = int(sum(val_vals))

    # Calculate percentages
    total_train = sum(train_vals)
    total_val = sum(val_vals)
    train_percents = [x / total_train * 100 for x in train_vals]
    val_percents = [x / total_val * 100 for x in val_vals]

    # Adjust figure size and plot layout to better accommodate long labels
    fig, ax = plt.subplots(figsize=(15, 10))  # Increase width for more horizontal space
    x = range(len(categories))
    train_bars = ax.bar([i - 0.2 for i in x], train_percents, width=0.4, label=f'Train {total_train_bboxes} Bboxes', align='center')
    val_bars = ax.bar([i + 0.2 for i in x], val_percents, width=0.4, label=f'Val {total_val_bboxes} Bboxes', align='center')
    ax.set_xlabel('Category ID')
    ax.set_ylabel('Percentage')
    ax.set_title('Distribution of BBox Categories in Train vs. Val')
    ax.set_xticks(x)
    ax.set_xticklabels(categories_names, rotation=75)  # Rotate labels to vertical for better fit
    ax.legend()

    # Ensure there is enough space between bars and text annotations
    # Adding counts above bars, with extra height adjustments
    for bars, values, is_train in zip([train_bars, val_bars], [train_vals, val_vals], [True, False]):
        for bar, value in zip(bars, values):
            yval = bar.get_height()
            if is_train:
                offset = .55  # Adjust the offset for train bar annotations
            else:
                offset = 0.05  # Smaller or no offset for validation bar annotations
            ax.text(bar.get_x() + bar.get_width()/2, yval + offset, f'{int(value)}', va='bottom', ha='center')  # Center align text

    plt.tight_layout()  # Adjust layout to make room for label rotation
    plt.savefig(f'{safety_set}/{safety_set}_category_distribution.png', bbox_inches='tight')

def move_images(data_set, folder):

    # Create the destination folder if it doesn't exist
    dest_folder = f'data/{folder}_images'
    os.makedirs(dest_folder, exist_ok=True)

    # Get the list of image IDs to move
    image_ids = set(image['file_name'] for image in data_set['images'])

    # Define supported image extensions
    image_extensions = ['.jpg', '.png', '.jpeg']

    # Move all the images to a single folder
    for path in tqdm(glob.glob('data/annotations/**/*', recursive=True), desc=f"Moving Images to {folder} Folder"):
        if os.path.isfile(path) and path.lower().endswith(tuple(image_extensions)):
            image_id = os.path.basename(path)
            if image_id in image_ids:
                shutil.copy(path, dest_folder)

def gather_annotations(safety_set):
    coco_annotations = {'images': [], 'annotations': [], 'categories': []}
    max_image_id = 0
    max_annotation_id = 0
    image_id_set = set()
    image_filename_set = set()

    for batch_folder in tqdm(glob.glob(f'data/annotations/{safety_set}/*'), desc="Gathering Annotations"):
        annotation_files = glob.glob(f'{batch_folder}/annotations/*')
        for annotation_file in annotation_files:
            old_id_to_new_id = {}
            
            with open(annotation_file) as f:
                annotations = json.load(f)
                
                # First loop through and make sure any leading folders are removed from the image file names
                for image in annotations['images']:
                    image['file_name'] = image['file_name'].split('/')[-1]

                # Update image IDs
                for image in annotations['images']:
                    old_id = image['id']
                    new_id = max_image_id
                    if new_id in image_id_set:
                        print(f"Image ID {new_id} already exists in the mapping")
                        continue
                    if image['file_name'] in image_filename_set:
                        continue
                    else:
                        image_filename_set.add(image['file_name'])
                        old_id_to_new_id[old_id] = new_id
                        image['id'] = new_id
                        max_image_id += 1

                    # Append the image to the total annotations
                    coco_annotations['images'].append(image)
                
                # Update annotation IDs
                for annotation in annotations['annotations']:
                    if annotation['image_id'] not in old_id_to_new_id:
                        continue
                    annotation['image_id'] = old_id_to_new_id[annotation['image_id']]
                    annotation['id'] = max_annotation_id
                    max_annotation_id += 1
                    
                    # Append the annotation to the total annotations
                    coco_annotations['annotations'].append(annotation)

                # Append the categories to the total annotations
                coco_annotations['categories'].extend(annotations['categories'])

    # Ensure the categories are unique
    unique_categories = {category['id']: category for category in coco_annotations['categories']}
    coco_annotations['categories'] = list(unique_categories.values())

    # Save the aggregated annotations to a JSON file
    with open(f'data/annotations/{safety_set}_total.json', 'w') as f:
        json.dump(coco_annotations, f, indent=4, sort_keys=True)

    # Ensure that each of the image_ids in the annotations are unique to a image_filename
    image_filenames = set()
    image_ids = set()
    for image in coco_annotations['images']:
        image_id = image['id']
        image_filename = image['file_name']
        if image_id in image_ids:
            print(f"Image ID {image_id} is not unique")
        elif image_filename in image_filenames:
            print(f"Image filename {image_filename} is not unique")
        else:
            image_ids.add(image_id)
            image_filenames.add(image_filename)

    return coco_annotations

if __name__ == '__main__':

    # Parse the safety set arguments
    parser = argparse.ArgumentParser(description='Divide the data into train and val sets')
    parser.add_argument('--safety_set', default='gh')
    parser.add_argument('--split_ratio', default=0.90)
    parser.add_argument('--images_annotation_split', default='images', help='Either images or annotations deciding the split type')
    args = parser.parse_args()

    # Set the safety set
    safety_set = str(args.safety_set)
    split_ratio = float(args.split_ratio)
    images_annotation_split = str(args.images_annotation_split)

    main(safety_set, split_ratio, images_annotation_split)








