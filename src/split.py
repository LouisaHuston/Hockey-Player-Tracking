import json
import os
import random

def split_coco_json(annotations_path, output_dir, train_file_name='train_annotations.json', test_file_name='test_annotations.json', split_ratio=0.8):
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    random.shuffle(images)
    split_index = int(len(images) * split_ratio)

    train_images = images[:split_index]
    test_images = images[split_index:]

    train_image_ids = set([img['id'] for img in train_images])
    test_image_ids = set([img['id'] for img in test_images])

    train_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in train_image_ids]
    test_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in test_image_ids]

    train_data = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': coco_data['categories']
    }
    test_data = {
        'images': test_images,
        'annotations': test_annotations,
        'categories': coco_data['categories']
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, train_file_name), 'w') as f:
        json.dump(train_data, f)
    with open(os.path.join(output_dir, test_file_name), 'w') as f:
        json.dump(test_data, f)
