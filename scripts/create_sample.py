import os
import shutil
import random

source_dirs = {
    "raw/ct": 5,
    "raw/xray": 5,
    "processed/ct": 5,
    "processed/xray": 5,
    "tampered/ct_baseline": 5,
    "tampered/ct_harder": 5,
    "tampered/xray": 5
}

base_dir = "data"
sample_dir = os.path.join(base_dir, "sample")

os.makedirs(sample_dir, exist_ok=True)

for rel_path, count in source_dirs.items():
    src_dir = os.path.join(base_dir, rel_path)
    dst_dir = os.path.join(sample_dir, rel_path)
    
    if os.path.exists(src_dir):
        all_files = []
        for root, _, files in os.walk(src_dir):
            for f in files:
                all_files.append(os.path.join(root, f))
        
        # Pick up to 'count' files randomly
        selected_files = random.sample(all_files, min(count, len(all_files)))
        
        for file_path in selected_files:
            # Flatten the directory structure in sample dir, or preserve it?
            # Easiest is just to copy into dst_dir and prefix with a random hash or original name if unique
            filename = os.path.basename(file_path)
            # Add parent dir name to avoid collisions
            parent_name = os.path.basename(os.path.dirname(file_path))
            new_filename = f"{parent_name}_{filename}" if parent_name not in rel_path else filename
            
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(file_path, os.path.join(dst_dir, new_filename))
        
        print(f"Copied {len(selected_files)} files to {dst_dir}")
    else:
        print(f"Source {src_dir} not found.")

print("Sample creation complete.")
