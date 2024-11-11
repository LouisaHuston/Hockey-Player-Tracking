import json
import os
import random
import shutil

def split_coco_json(annotations_path, output_dir, image_dir, train_file_name='train_annotations.json', test_file_name='test_annotations.json', split_ratio=0.8):
    # Load COCO annotations
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)

    # Shuffle images and split into train and test
    images = coco_data['images']
    random.shuffle(images)
    split_index = int(len(images) * split_ratio)

    train_images = images[:split_index]
    test_images = images[split_index:]

    # Extract image IDs for each split
    train_image_ids = set(img['id'] for img in train_images)
    test_image_ids = set(img['id'] for img in test_images)

    # Update image file names in JSON data to replace '/' with '_'
    for img in train_images + test_images:
        img['file_name'] = img['file_name'].replace('/', '_')

    # Filter annotations by image IDs
    train_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in train_image_ids]
    test_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in test_image_ids]

    # Organize output data
    train_data = {'images': train_images, 'annotations': train_annotations, 'categories': coco_data['categories']}
    test_data = {'images': test_images, 'annotations': test_annotations, 'categories': coco_data['categories']}

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    train_image_dir = os.path.join(image_dir, 'train')
    test_image_dir = os.path.join(image_dir, 'test')
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(test_image_dir, exist_ok=True)

    # Save split annotations with updated file names
    with open(os.path.join(output_dir, train_file_name), 'w') as f:
        json.dump(train_data, f)
    with open(os.path.join(output_dir, test_file_name), 'w') as f:
        json.dump(test_data, f)

    # Move images to train and test directories with updated file names
    for img in train_images:
        img_file = img['file_name']
        src_path = os.path.join(image_dir, img_file.replace('_', '/'))
        dst_path = os.path.join(train_image_dir, img_file)
        shutil.copy2(src_path, dst_path)

    for img in test_images:
        img_file = img['file_name']
        src_path = os.path.join(image_dir, img_file.replace('_', '/'))
        dst_path = os.path.join(test_image_dir, img_file)
        shutil.copy2(src_path, dst_path)
