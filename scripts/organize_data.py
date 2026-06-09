import os
import shutil

data_dir = "data"
os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "tampered"), exist_ok=True)

# Define movements
moves = {
    "dicom_raw": "raw/ct",
    "dicom_raw_xray": "raw/xray",
    "dicom_processed": "processed/ct",
    "dicom_processed_xray": "processed/xray",
    "tampered_dataset": "tampered/ct_baseline",
    "tampered_dataset_harder": "tampered/ct_harder",
    "tampered_dataset_xray": "tampered/xray",
}

for src_name, dst_name in moves.items():
    src_path = os.path.join(data_dir, src_name)
    dst_path = os.path.join(data_dir, dst_name)
    if os.path.exists(src_path):
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.move(src_path, dst_path)
        print(f"Moved {src_name} to {dst_name}")

if os.path.exists(os.path.join(data_dir, "synthetic")):
    shutil.rmtree(os.path.join(data_dir, "synthetic"))
    print("Removed synthetic")

print("Dataset organization complete.")
