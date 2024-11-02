

from src.download_data import download_data
from src.process import extract_frames, extract_annotations, create_coco_annotations

def main():
    # Download data
    download_data("https://github.com/grant81/hockeyTrackingDataset.git", "data/hockeyTrackingDataset")

    # Extract frames from video clips
    root_folder = "data/hockeyTrackingDataset/clips"
    extract_frames(root_folder)

    # Extract and process annotations
    test_annotations = extract_annotations("data/hockeyTrackingDataset/MOT_Challenge_Sytle_Label/test/")
    train_annotations = extract_annotations("data/hockeyTrackingDataset/MOT_Challenge_Sytle_Label/train/")
    annotations_array = test_annotations + train_annotations

    # Create COCO-style annotations
    images_folder = "data/hockeyTrackingDataset/images/"
    coco_data = create_coco_annotations(annotations_array, images_folder)

    # Save COCO annotations to JSON
    with open('coco_annotations.json', 'w') as f:
        json.dump(coco_data, f)

    print("COCO JSON generated and saved as 'coco_annotations.json'.")

if __name__ == "__main__":
    main()
