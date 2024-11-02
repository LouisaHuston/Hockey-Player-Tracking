import os
import cv2
import json
from tqdm import tqdm

def extract_frames(root_folder):
    """Find and extract frames from all .mp4 files in the given root folder."""
    video_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith(".mp4"):
                video_files.append(os.path.join(dirpath, file))

    for video_path in video_files:
        relative_path = os.path.relpath(os.path.dirname(video_path), root_folder)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_folder = os.path.join('data/images', relative_path, video_name)

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_number = 0

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        with tqdm(total=frame_count, desc=f"Extracting frames from {os.path.basename(video_path)}", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_filename = f"{output_folder}/{frame_number}.jpg"
                cv2.imwrite(frame_filename, frame)
                frame_number += 1
                pbar.update(1)

        cap.release()

def extract_annotations(root_folder):
    """Extract annotation data from .txt files in the given folder."""
    gt_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith(".txt"):
                gt_files.append(os.path.join(dirpath, file))

    annotations_array = []
    for gt_path in tqdm(gt_files, desc="Processing text files", unit="file"):
        with open(gt_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                frame_number, player_id, x1, y1, w, h, confidence, tracklet_id, visibility = line.strip().split(',')
                annotation = {
                    "frame_number": int(frame_number),
                    "player_id": int(player_id),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "confidence": float(confidence),
                    "tracklet_id": int(tracklet_id),
                    "visibility": float(visibility),
                    "file_path": gt_path
                }
                annotations_array.append(annotation)

    return annotations_array

def create_coco_annotations(annotations, images_folder):
    """Generate COCO-style JSON annotations."""
    # (Implement logic from the original function here)
