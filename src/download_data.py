import os
import subprocess

def download_dataset(dataset_dir):

    if not os.path.exists(dataset_dir):
        subprocess.run(['git', 'clone', 'https://github.com/grant81/hockeyTrackingDataset.git'])
        print("Dataset downloaded.")
    else:
        print("Dataset already downloaded.")
