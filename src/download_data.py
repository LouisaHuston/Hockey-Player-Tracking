
def download_hockey_dataset():
    # Check to see if the data is in the right directory, if not clone repo and move data
    import os
    import subprocess
    import cv2
    import logging
    from tqdm import tqdm
    
    # Define the dataset repository and local directory
    repo_url = "https://github.com/grant81/hockeyTrackingDataset.git"
    local_dir = "hockeyTrackingDataset"
    
    # Check if the dataset directory exists
    if not os.path.exists(local_dir):
        subprocess.run(["git", "clone", repo_url])
    
    # Change to the dataset directory
    os.chdir(local_dir)
    
    def extract_frames(video_path, output_folder):
        """Extract frames from a video and save them as images."""
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_number = 0
    
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
        # Add a progress bar
        with tqdm(total=frame_count, desc=f"Extracting frames from {os.path.basename(video_path)}", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
    
                # Save frame as JPG file
                frame_filename = f"{output_folder}/{frame_number}.jpg"
                cv2.imwrite(frame_filename, frame)
                frame_number += 1
    
                # Update the progress bar
                pbar.update(1)
    
        cap.release()
    
        # Print completion message
        print(f"Completed extracting frames from {os.path.basename(video_path)} into {output_folder}")
    
    def find_videos_and_extract_frames(root_folder):
        """Find all .mp4 files in the root_folder and extract their frames."""
        video_files = []
    
        # Collect all video file paths
        for dirpath, _, filenames in os.walk(root_folder):
            for file in filenames:
                if file.endswith(".mp4"):
                    video_files.append(os.path.join(dirpath, file))
    
        # Process each video file
        for video_path in video_files:
            # Generate output path based on the video path
            relative_path = os.path.relpath(os.path.dirname(video_path), root_folder)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_folder = os.path.join('images', relative_path, video_name)
    
            # Extract frames
            extract_frames(video_path, output_folder)
    
            # This will limit to ONE VIDEO
            return
