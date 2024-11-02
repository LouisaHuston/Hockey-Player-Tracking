import os

def download_data(repo_url, output_dir):
    """Clone a Git repository to the specified directory."""
    if not os.path.exists(output_dir):
        os.system(f"git clone {repo_url} {output_dir}")
    else:
        print("Data already exists.")
