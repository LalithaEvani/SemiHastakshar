#!/usr/bin/env python3
"""
Creates an LMDB dataset for inference by recursively scanning a directory of images.

This script requires no ground truth file. It finds all image files in a given
folder and its subfolders. For each image, it creates an LMDB entry with a
hardcoded dummy label (e.g., "0") and saves the original filename.

This is ideal for preparing an unlabeled test set for a model, where predictions
can later be matched to the original files via the saved image names.
"""
import io
import os
import argparse
import lmdb
import numpy as np
from PIL import Image
from tqdm import tqdm

def checkImageIsValid(imageBin):
    """Checks if the image binary data is valid and can be opened."""
    if imageBin is None:
        return False
    try:
        img = Image.open(io.BytesIO(imageBin)).convert('RGB')
        return np.prod(img.size) > 0
    except (IOError, OSError):
        return False

def writeCache(env, cache):
    """Writes the cache to the LMDB transaction."""
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def createDataset(inputPath, outputPath, checkValid=True):
    """
    Create an LMDB dataset by scanning image folders for inference.

    Args:
        inputPath (str): Root folder path containing the image (sub)folders.
        outputPath (str): Path to the folder where the LMDB will be created.
        checkValid (bool): If true, check the validity of every image.
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    
    cache = {}
    cnt = 1

    # Define valid image extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')
    hardcoded_label = "0"

    # --- Step 1: Discover all image files recursively ---
    print(f"Scanning for images in: {inputPath}")
    all_image_paths = []
    for dirpath, _, filenames in os.walk(inputPath):
        for filename in filenames:
            if filename.lower().endswith(image_extensions):
                full_path = os.path.join(dirpath, filename)
                all_image_paths.append(full_path)
    
    nSamples = len(all_image_paths)
    if nSamples == 0:
        print(f"Error: No images found in {inputPath}. Please check the path and file extensions.")
        env.close()
        return
        
    print(f"Found {nSamples} images. Starting conversion to LMDB...")

    # --- Step 2: Iterate through discovered images and create dataset ---
    for full_image_path in tqdm(all_image_paths, desc='Converting to LMDB'):
        image_name = os.path.basename(full_image_path)
        
        # The label is hardcoded to "0" for inference purposes
        label = hardcoded_label

        with open(full_image_path, 'rb') as f:
            imageBin = f.read()

        if checkValid and not checkImageIsValid(imageBin):
            error_msg = f"Skipping invalid or corrupted image: {full_image_path}"
            print(error_msg)
            with open(os.path.join(outputPath, 'error_image_log.txt'), 'a', encoding='utf-8') as log:
                log.write(error_msg + '\n')
            continue

        # Create keys for the database
        imageKey = f'image-{cnt:09d}'.encode()
        labelKey = f'label-{cnt:09d}'.encode()
        nameKey = f'image_name-{cnt:09d}'.encode()

        # Add data to cache
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode('utf-8') # Use the hardcoded label
        cache[nameKey] = image_name.encode('utf-8') # Store the original filename

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}

        cnt += 1

    if cache:
        writeCache(env, cache)

    nSamples_written = cnt - 1
    with env.begin(write=True) as txn:
        txn.put('num-samples'.encode(), str(nSamples_written).encode())
    
    env.close()
    print(f"\nSuccessfully created dataset with {nSamples_written} samples at: {outputPath}")
    print(f"All entries have a default label of '{hardcoded_label}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create an LMDB for inference from image folders with a dummy label.")
    parser.add_argument("--inputPath", type=str, required=True,
                        help="Path to the root folder containing image (sub)folders.")
    parser.add_argument("--outputPath", type=str, required=True,
                        help="Path to the folder where the LMDB dataset will be created.")

    args = parser.parse_args()
    
    createDataset(args.inputPath, args.outputPath)

'''
    python 6_img_lmdb_imgnames.py \
      --inputPath /ssd_scratch/cvit/lalitha/unlabelled_bengali/train_lmdb/ \
      --outputPath /ssd_scratch/cvit/lalitha/unlabelled_bengali/train/ 
python 6_img_lmdb_imgnames.py \
      --inputPath /ssd_scratch/cvit/lalitha/unlabelled/train \
      --outputPath /ssd_scratch/cvit/lalitha/unlabelled/train_lmdb/ 
'''