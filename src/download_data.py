# src/download_data.py
import os
import subprocess

def download_dataset(repo_url, destination_path):
    """
    Downloads the dataset from the given GitHub repository.
    """
    if not os.path.exists(destination_path):
        print(f"Cloning repository from {repo_url}...")
        subprocess.run(['git', 'clone', repo_url, destination_path])
        print("Dataset downloaded successfully.")
    else:
        print("Dataset already exists. Skipping download.")
