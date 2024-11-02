# src/download_data.py

def download_dataset():
    import os
    import subprocess

    dataset_dir = 'hockeyTrackingDataset'

    if not os.path.exists(dataset_dir):
        subprocess.run(['git', 'clone', 'https://github.com/grant81/hockeyTrackingDataset.git'])
        print("Dataset downloaded.")
    else:
        print("Dataset already downloaded.")

    return os.path.abspath(dataset_dir)
