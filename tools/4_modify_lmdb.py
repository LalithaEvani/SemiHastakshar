#!/usr/bin/env python3
"""
Script to read an existing LMDB dataset (with image and label keys)
and create a new LMDB dataset adding an 'imgname' key for each entry,
using the entry's index as the image name value.
"""
import os
import lmdb
import argparse
from tqdm import tqdm
import sys # For exiting on error

# Re-using the robust writeCache function from the original script
def writeCache(env, cache):
    """Writes the cache dictionary to the LMDB environment."""
    if not cache:
        return
    try:
        with env.begin(write=True) as txn:
            for k, v in cache.items():
                # Ensure key and value are bytes
                if isinstance(k, str):
                    k = k.encode('utf-8')
                if isinstance(v, str):
                    v = v.encode('utf-8')
                elif not isinstance(v, bytes):
                    try:
                        v = str(v).encode('utf-8')
                    except Exception as e:
                        print(f"Error encoding value for key {k.decode('utf-8', 'ignore')}: {e}. Value type: {type(v)}. Skipping.", file=sys.stderr)
                        continue # Skip this key-value pair

                try:
                    txn.put(k, v)
                except lmdb.BadValsizeError:
                     print(f"Error: Value too large for key {k.decode('utf-8', 'ignore')}. Size: {len(v)}. Skipping.", file=sys.stderr)
                except Exception as e:
                     print(f"Error putting key {k.decode('utf-8', 'ignore')}: {e}. Skipping.", file=sys.stderr)
    except Exception as e:
        print(f"Error opening transaction to write cache: {e}", file=sys.stderr)

def modify_lmdb_add_imgname(inputLmdbPath, outputLmdbPath):
    """
    Reads an LMDB dataset, adds an 'imgname' key derived from the index,
    and writes to a new LMDB dataset.

    ARGS:
        inputLmdbPath : Path to the existing LMDB directory.
        outputLmdbPath: Path to the new LMDB directory to be created.
    """
    if not os.path.isdir(inputLmdbPath):
        print(f"Error: Input LMDB path not found or not a directory: {inputLmdbPath}", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(outputLmdbPath):
         # Basic check to prevent accidental overwrite of input
         if os.path.realpath(inputLmdbPath) == os.path.realpath(outputLmdbPath):
              print(f"Error: Output path cannot be the same as the input path.", file=sys.stderr)
              sys.exit(1)
         print(f"Warning: Output directory '{outputLmdbPath}' already exists. Files may be overwritten.")
    else:
        os.makedirs(outputLmdbPath, exist_ok=True)

    # Estimate map size (start with input size, add some buffer)
    # This is a rough estimate, might need adjustment for very large datasets
    try:
        input_stat = os.stat(os.path.join(inputLmdbPath, 'data.mdb'))
        map_size = input_stat.st_size * 1.5 # Add 50% buffer
        # Ensure minimum size and reasonable maximum if calculation is off
        map_size = max(map_size, 1024**3) # Min 1 GB
        map_size = min(map_size, 1099511627776) # Max 1 TB (like original script)
        print(f"Estimated map size for output LMDB: {map_size / (1024**3):.2f} GB")
    except FileNotFoundError:
         print(f"Warning: data.mdb not found in {inputLmdbPath}. Using default map size (1TB).")
         map_size = 1099511627776 # Default 1TB
    except Exception as e:
         print(f"Warning: Could not estimate input size ({e}). Using default map size (1TB).")
         map_size = 1099511627776 # Default 1TB


    try:
        env_in = lmdb.open(inputLmdbPath, readonly=True, lock=False, readahead=False, meminit=False)
        env_out = lmdb.open(outputLmdbPath, map_size=int(map_size)) # Use estimated map size
    except lmdb.Error as e:
        print(f"Error opening LMDB environment: {e}", file=sys.stderr)
        sys.exit(1)

    cache = {}
    cnt = 0 # Counter for processed samples
    num_samples_in = 0

    with env_in.begin(write=False) as txn_in:
        try:
            num_samples_raw = txn_in.get(b'num-samples')
            if num_samples_raw:
                num_samples_in = int(num_samples_raw.decode('utf-8'))
                print(f"Input LMDB reports {num_samples_in} samples.")
            else:
                print("Warning: 'num-samples' key not found in input LMDB. Will count entries.")
                # Estimate from number of keys / 2 (image + label)
                num_samples_in = txn_in.stat()['entries'] // 2
                print(f"Estimated samples based on key count: {num_samples_in}")
        except Exception as e:
             print(f"Warning: Could not read or parse 'num-samples' key: {e}. Will count entries.")
             num_samples_in = txn_in.stat()['entries'] // 2
             print(f"Estimated samples based on key count: {num_samples_in}")


        cursor = txn_in.cursor()

        print(f"Processing entries from '{os.path.basename(inputLmdbPath)}'...")
        # Use num_samples_in for tqdm total if available and seems reasonable, otherwise don't set total
        tqdm_total = num_samples_in if num_samples_in > 0 else None

        for key, value in tqdm(cursor, total=tqdm_total, desc=f'Converting to {os.path.basename(outputLmdbPath)}'):
            try:
                key_str = key.decode('utf-8')
            except UnicodeDecodeError:
                print(f"Warning: Skipping key that is not valid UTF-8: {key}", file=sys.stderr)
                continue

            # Process only 'image-*' keys to avoid duplicates and ensure pairing
            if key_str.startswith('image-'):
                try:
                    # 1. Extract index
                    parts = key_str.split('-')
                    if len(parts) != 2:
                        print(f"Warning: Skipping malformed key: {key_str}", file=sys.stderr)
                        continue
                    index_str = parts[1]
                    # Validate index format (optional, but good practice)
                    if not index_str.isdigit():
                         print(f"Warning: Skipping key with non-numeric index: {key_str}", file=sys.stderr)
                         continue

                    # 2. Get corresponding label value
                    labelKey = f'label-{index_str}'.encode('utf-8')
                    labelValue = txn_in.get(labelKey)

                    if labelValue is None:
                        print(f"Warning: Label key '{labelKey.decode()}' not found for image key '{key_str}'. Skipping entry.", file=sys.stderr)
                        continue

                    # 3. Create image name key and value
                    imgNameKey = f'image_name-{index_str}'.encode('utf-8')
                    imgNameValue = index_str.encode('utf-8') # Value is the index itself

                    # 4. Add all three to cache
                    cache[key] = value         # imageKey: imageBin
                    cache[labelKey] = labelValue # labelKey: labelStr
                    cache[imgNameKey] = imgNameValue # imgNameKey: indexStr

                    cnt += 1

                    # 5. Write cache periodically
                    if cnt % 1000 == 0:
                        writeCache(env_out, cache)
                        cache = {}
                        # print(f'Written {cnt} entries...') # Reduce verbosity

                except Exception as e:
                    print(f"Error processing key {key_str}: {e}", file=sys.stderr)
                    # Decide whether to continue or stop on error
                    # continue

    # Write remaining cache
    if cache:
        writeCache(env_out, cache)

    # Write the final 'num-samples' key using the count of processed entries
    cache = {} # Clear cache before writing metadata
    cache['num-samples'.encode('utf-8')] = str(cnt).encode('utf-8')
    writeCache(env_out, cache)

    env_in.close()
    env_out.close()

    print(f'\nFinished creating dataset "{os.path.basename(outputLmdbPath)}" with {cnt} samples at {outputLmdbPath}')
    if num_samples_in > 0 and cnt != num_samples_in:
         print(f"Warning: Number of samples processed ({cnt}) differs from input 'num-samples' ({num_samples_in}). Check input LMDB or logs for skipped entries.", file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create a new LMDB dataset from an existing one, adding an 'imgname' key based on the entry index.",
        formatter_class=argparse.RawTextHelpFormatter # Keep newlines in help
        )
    parser.add_argument("--inputLmdbPath", type=str, required=True,
                        help="Path to the source LMDB directory (containing image-* and label-* keys).")
    parser.add_argument("--outputLmdbPath", type=str, required=True,
                        help="Path to the destination LMDB directory to be created.\n"
                             "It will contain image-*, label-*, and imgname-* keys.")

    args = parser.parse_args()

    print(f"Input LMDB:  {args.inputLmdbPath}")
    print(f"Output LMDB: {args.outputLmdbPath}")
    print("-" * 20)

    modify_lmdb_add_imgname(args.inputLmdbPath, args.outputLmdbPath)

    print("Done.")

# Example usage:
# python modify_lmdb.py --inputLmdbPath /path/to/existing_lmdb --outputLmdbPath /path/to/new_lmdb_with_names
# python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/data/val --outputLmdbPath /ssd_scratch/cvit/lalitha/data/val_new
# python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/2_pdf_scraping_data/unlabelled_one/500_sampled_test_data/lmdb_data --outputLmdbPath /ssd_scratch/cvit/lalitha/2_pdf_scraping_data/unlabelled_one/500_sampled_test_data/new_lmdb_data/
'''
python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/data/val \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/data/val_new

python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/data/train/real \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/data/train/real_new

python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/data/test/iiit-indic-hw-words \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/data/test/iiit-indic-hw-words_new

python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/data/test/crowd_sourced \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/data/test/crowd_sourced_new
    
python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/data/train/real/sampled/ \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/data/train/real/sampled_new

python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/data/telugu/uc/test \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/data/telugu/uc/test_new/

python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/data/telugu/words/test \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/data/telugu/words/test_new/

python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/data/test/iiit-indic-hw-uc \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/data/test/iiit-indic-hw-uc-new
      
'''