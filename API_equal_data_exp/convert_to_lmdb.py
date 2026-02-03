#!/usr/bin/env python3
""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """
import io
import os
from tqdm import tqdm
import lmdb
import numpy as np
from PIL import Image
import argparse
import ntpath # For path manipulation

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    try:
        img = Image.open(io.BytesIO(imageBin)).convert('RGB')
        return np.prod(img.size) > 0
    except IOError:
        return False
    except Exception: # Catch other PIL errors
        return False


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            # Ensure value is bytes
            if isinstance(v, str):
                v = v.encode()
            elif not isinstance(v, bytes):
                 # Attempt to convert common types, handle others as errors or log
                 try:
                     v = str(v).encode()
                 except Exception as e:
                     print(f"Error encoding value for key {k}: {e}. Value type: {type(v)}")
                     continue # Skip this key-value pair
            try:
                txn.put(k, v)
            except lmdb.BadValsizeError:
                 print(f"Error: Value too large for key {k}. Size: {len(v)}")
            except Exception as e:
                 print(f"Error putting key {k}: {e}")


def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation including image names.
    ARGS:
        inputPath  : input folder path where image files are located (e.g., ./images/)
        outputPath : LMDB output path
        gtFile     : list of image filename (relative to inputPath) and label (e.g., image1.png\tlabel1)
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776) # 1TB map size

    cache = {}
    cnt = 1

    try:
        with open(gtFile, 'r', encoding='utf-8') as f:
            data = f.readlines()
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {gtFile}")
        return
    except Exception as e:
        print(f"Error reading ground truth file {gtFile}: {e}")
        return

    nSamples = len(data)
    error_log_path = os.path.join(outputPath, 'error_image_log.txt')
    if os.path.exists(error_log_path):
        os.remove(error_log_path) # Clear log on new run

    print(f"Processing {nSamples} samples from {gtFile}...")
    for i, line in enumerate(tqdm(data, desc=f'Converting to LMDB ({os.path.basename(outputPath)})')):
        line = line.strip()
        if not line:
            print(f"Warning: Skipped empty line at index {i} in {gtFile}")
            continue

        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            print(f"Warning: Skipped malformed line at index {i} in {gtFile}: '{line}'")
            with open(error_log_path, 'a', encoding='utf-8') as log:
                 log.write(f"Malformed line {i}: {line}\n")
            continue

        imageRelPath, label = parts
        # ntpath.basename ensures we get the filename even with mixed separators
        imageName = ntpath.basename(imageRelPath)
        imagePath = os.path.join(inputPath, imageRelPath) # inputPath is the root for image files listed in gtFile

        if not os.path.exists(imagePath):
            print(f"Warning: Image file not found: {imagePath} (referenced in {gtFile})")
            with open(error_log_path, 'a', encoding='utf-8') as log:
                 log.write(f"Image not found {i}: {imagePath}\n")
            continue

        try:
            with open(imagePath, 'rb') as f:
                imageBin = f.read()
        except Exception as e:
            print(f"Error reading image file {imagePath}: {e}")
            with open(error_log_path, 'a', encoding='utf-8') as log:
                 log.write(f"Error reading image {i}: {imagePath}, {e}\n")
            continue

        if checkValid and not checkImageIsValid(imageBin):
            msg = f'{i}-th image data is not valid: {imagePath}'
            print(f"Warning: {msg}")
            with open(error_log_path, 'a', encoding='utf-8') as log:
                log.write(f"{msg}\n")
            continue

        imageKey = f'image-{cnt:09d}'.encode('utf-8')
        labelKey = f'label-{cnt:09d}'.encode('utf-8')
        imgNameKey = f'image_name-{cnt:09d}'.encode('utf-8') # New key for image name

        cache[imageKey] = imageBin
        cache[labelKey] = label.encode('utf-8')
        cache[imgNameKey] = imageName.encode('utf-8') # Store the image name

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            # print('Written %d / %d' % (cnt, nSamples)) # Reduce verbosity inside loop
        cnt += 1

    # Write remaining cache
    if cache:
        writeCache(env, cache)

    actual_nSamples = cnt - 1
    cache = {} # Clear cache before writing metadata
    cache['num-samples'.encode('utf-8')] = str(actual_nSamples).encode('utf-8')
    writeCache(env, cache)
    env.close()
    print(f'Created dataset "{os.path.basename(outputPath)}" with {actual_nSamples} samples at {outputPath}')

    # Report errors if any
    if os.path.exists(error_log_path):
        with open(error_log_path, 'r', encoding='utf-8') as log:
            errors = log.readlines()
        if errors:
            print(f"Warning: Encountered {len(errors)} issues during LMDB creation. See log: {error_log_path}")
        else:
            os.remove(error_log_path) # Remove empty log file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create LMDB dataset including image names.")
    parser.add_argument("--inputPath", type=str, required=True, help="Path to the input image folder (root directory containing the images referenced in gtFile).")
    parser.add_argument("--gtFile", type=str, required=True, help='Path to the ground truth file (format: relative_image_path\\tlabel).')
    parser.add_argument("--outputPath", type=str, required=True, help="Path to the output LMDB directory.")
    parser.add_argument('--checkValid', action='store_true', help='Perform image validity checks.')

    args = parser.parse_args()
    createDataset(args.inputPath, args.gtFile, args.outputPath, args.checkValid)

    # Example usage:
    # python convert_to_lmdb_with_names.py --inputPath /path/to/images --gtFile /path/to/gt.txt --outputPath /path/to/output_lmdb --checkValid