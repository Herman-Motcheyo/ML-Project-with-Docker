import os
import subprocess
from datetime import datetime
from app.utils import load_params

UPLOAD_FOLDER = load_params()["UPLOAD_FOLDER"]

def save_file_and_push_to_dvc(file_bytes, filename: str):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    subprocess.run(["dvc", "add", file_path], check=True)
    subprocess.run(["git", "add", f"{file_path}.dvc"], check=True)
    subprocess.run(["git", "commit", "-m", f"Add {filename} via UI at {datetime.now()}"], check=True)
    subprocess.run(["dvc", "push"], check=True)

    return file_path
