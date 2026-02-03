import os
import lmdb
import numpy as np
from PIL import Image
import argparse
import io
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# --- Helper functions (from previous versions) ---
def sanitize_filename(name_str, default_ext=".png"):
    if not name_str: return None
    base_name = os.path.basename(name_str)
    name_part, ext_part = os.path.splitext(base_name)
    sanitized_name_part = re.sub(r'[^\w\s.-]', '_', name_part).strip()
    sanitized_name_part = re.sub(r'\s+', '_', sanitized_name_part)
    if not sanitized_name_part: return None
    if ext_part and ext_part.startswith('.') and len(ext_part) > 1 and len(ext_part) < 6 and ext_part[1:].isalnum():
        sanitized_ext_part = ext_part.lower()
    else:
        sanitized_ext_part = default_ext
    return sanitized_name_part + sanitized_ext_part

def write_cache_to_lmdb(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

# --- Core Logic ---

def create_new_lmdb_with_guaranteed_name_key(
    input_lmdb_path,
    output_new_lmdb_path, # Path for the NEW LMDB
    custom_key_title="imgNameKey",
    default_img_ext=".png",
    num_workers=4 # For reading the input LMDB if needed for speed, though writing is sequential
):
    """
    Reads an input LMDB and creates a NEW LMDB.
    The new LMDB will always have the 'custom_key_title-%09d' populated.
    If the key existed in the input, its value is used.
    If not, an index-based name (e.g., "000000001.png") is used as the value.
    """
    print(f"\n--- Creating New LMDB with Guaranteed Name Key ---")
    print(f"Input LMDB: {input_lmdb_path}")
    print(f"Output New LMDB: {output_new_lmdb_path}")
    print(f"Custom Key for Image Name: {custom_key_title}")

    if os.path.exists(output_new_lmdb_path):
        print(f"Warning: Output LMDB path '{output_new_lmdb_path}' already exists. It will be overwritten.")
        # For safety, you might want to add a prompt or an error here instead of automatic overwrite.
        # For now, let's assume it's okay to proceed by deleting it first.
        import shutil
        try:
            shutil.rmtree(output_new_lmdb_path)
        except Exception as e:
            print(f"Error removing existing output LMDB: {e}. Please remove it manually.")
            return False
            
    os.makedirs(output_new_lmdb_path, exist_ok=True)

    env_read = None
    env_write = None
    try:
        env_read = lmdb.open(input_lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        # Estimate map size for the new LMDB, can be same as original or slightly larger
        map_size_read = env_read.info().get('map_size', 1099511627776) # Default 1TB if not found
        env_write = lmdb.open(output_new_lmdb_path, map_size=int(map_size_read * 1.1)) # 10% larger
    except lmdb.Error as e:
        print(f"Error opening LMDB environments: {e}")
        if env_read: env_read.close()
        if env_write: env_write.close()
        return False

    new_lmdb_cache = {}
    entries_written = 0

    with env_read.begin(write=False) as txn_read:
        num_samples_bytes = txn_read.get(b'num-samples')
        if not num_samples_bytes:
            print(f"Error: 'num-samples' key not found in input LMDB '{input_lmdb_path}'.")
            env_read.close(); env_write.close(); return False
        try:
            num_samples = int(num_samples_bytes.decode())
        except ValueError:
            print(f"Error: Could not decode 'num-samples' from input LMDB.")
            env_read.close(); env_write.close(); return False

        if num_samples == 0:
            print("Input LMDB has 0 samples. Creating an empty new LMDB.")
            new_lmdb_cache[b'num-samples'] = b'0'
            write_cache_to_lmdb(env_write, new_lmdb_cache)
            env_read.close(); env_write.close(); return True
            
        print(f"Processing {num_samples} entries from '{input_lmdb_path}' to create '{output_new_lmdb_path}'.")

        for i in tqdm(range(1, num_samples + 1), desc="Building new LMDB"):
            idx_str = f"{i:09d}"
            img_key_orig = f"image-{idx_str}".encode()
            lbl_key_orig = f"label-{idx_str}".encode()
            custom_name_key_orig = f"{custom_key_title}-{idx_str}".encode() # Key to check in original
            
            # Keys for the new LMDB will be the same
            img_key_new = img_key_orig
            lbl_key_new = lbl_key_orig
            custom_name_key_new = custom_name_key_orig

            img_bin = txn_read.get(img_key_orig)
            lbl_bin = txn_read.get(lbl_key_orig)

            if not (img_bin and lbl_bin):
                print(f"Warning: Missing image or label for index {i} in input LMDB. Skipping.")
                continue

            image_name_for_new_lmdb_value = None

            # Check if custom name key exists in the original LMDB
            original_name_bytes = txn_read.get(custom_name_key_orig)
            if original_name_bytes:
                try:
                    # Use the existing name if present and decodable
                    image_name_for_new_lmdb_value = original_name_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    print(f"Warning: Could not decode name from '{custom_name_key_orig.decode()}' for index {i}. Using index-based name.")
                    # Fallthrough to generate index-based name
            
            # If not found or couldn't decode, generate an index-based name
            if not image_name_for_new_lmdb_value:
                image_name_for_new_lmdb_value = f"{idx_str}{default_img_ext}"
            
            # Add to cache for the new LMDB
            new_lmdb_cache[img_key_new] = img_bin
            new_lmdb_cache[lbl_key_new] = lbl_bin
            new_lmdb_cache[custom_name_key_new] = image_name_for_new_lmdb_value.encode('utf-8')
            entries_written += 1

            if len(new_lmdb_cache) >= 3000: # Approx 1000 entries (3 keys per entry)
                write_cache_to_lmdb(env_write, new_lmdb_cache)
                new_lmdb_cache = {}
        
    # Write any remaining cache and the num-samples for the new LMDB
    if new_lmdb_cache or entries_written == 0: # Ensure num-samples is written even if no actual image entries
        new_lmdb_cache[b'num-samples'] = str(entries_written).encode()
        write_cache_to_lmdb(env_write, new_lmdb_cache)
    elif entries_written > 0: # If cache was empty but entries were written
        with env_write.begin(write=True) as txn_w_final:
            txn_w_final.put(b'num-samples', str(entries_written).encode())


    env_read.close()
    env_write.close()
    print(f"New LMDB created at '{output_new_lmdb_path}' with {entries_written} samples.")
    return True


def extract_images_and_gt_from_processed_lmdb(
    processed_lmdb_path, # Path to the NEW LMDB created in the previous step
    output_image_dir,
    output_gt_filepath,
    custom_key_title="image_name", # This key is now guaranteed to exist
    num_workers=4,
    default_img_ext=".png" # Still needed for sanitize_filename
):
    """
    Extracts images and labels from the processed LMDB.
    It assumes 'custom_key_title-%09d' now exists for all entries.
    """
    print(f"\n--- Extracting Images and GT from Processed LMDB ---")
    print(f"Processed LMDB: {processed_lmdb_path}")
    print(f"Output Images: {output_image_dir}")
    print(f"Output GT File: {output_gt_filepath}")

    os.makedirs(output_image_dir, exist_ok=True)
    env_extract = None
    try:
        env_extract = lmdb.open(processed_lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    except lmdb.Error as e:
        print(f"Error opening processed LMDB at {processed_lmdb_path}: {e}")
        return False

    num_samples_extract = 0
    with env_extract.begin(write=False) as txn:
        num_samples_bytes = txn.get('num-samples'.encode())
        if not num_samples_bytes:
            print(f"Error: 'num-samples' key not found in processed LMDB: {processed_lmdb_path}")
            env_extract.close(); return False
        try: num_samples_extract = int(num_samples_bytes.decode())
        except ValueError:
            print(f"Error: Could not decode 'num-samples' from processed LMDB.")
            env_extract.close(); return False
            
    if num_samples_extract == 0:
        print("Processed LMDB has 0 samples. Nothing to extract."); env_extract.close(); return True

    extracted_label_entries = []
    
    def _extract_single_from_processed(idx, txn_ref, out_img_dir, cust_key_title, def_img_ext):
        img_key = f'image-{idx:09d}'.encode()
        lbl_key = f'label-{idx:09d}'.encode()
        # This key should now always exist in the processed LMDB
        img_name_key_in_lmdb = f'{cust_key_title}-{idx:09d}'.encode()

        img_bin = txn_ref.get(img_key)
        lbl_bin = txn_ref.get(lbl_key)
        img_name_val_bytes = txn_ref.get(img_name_key_in_lmdb) # Get the name stored in LMDB

        if not (img_bin and lbl_bin and img_name_val_bytes):
            print(f"Warning: Missing data for index {idx} in processed LMDB. Skipping.")
            return None
        
        try:
            image_name_from_lmdb = img_name_val_bytes.decode('utf-8')
        except UnicodeDecodeError:
            print(f"Warning: Could not decode image name for index {idx}. Using fallback.")
            image_name_from_lmdb = f"{idx:09d}{def_img_ext}" # Fallback if stored name is corrupt

        # Sanitize the name obtained from LMDB before using it as a filename
        output_filename = sanitize_filename(image_name_from_lmdb, default_ext=def_img_ext)
        if not output_filename: # If sanitization results in empty string
            output_filename = f"fallback_{idx:09d}{def_img_ext}"


        save_path = os.path.join(out_img_dir, output_filename)
        try:
            Image.open(io.BytesIO(img_bin)).save(save_path)
            label_text = lbl_bin.decode('utf-8')
            return f"{output_filename}\t{label_text}"
        except Exception as e:
            print(f"Error during extraction of index {idx} (img: {output_filename}): {e}")
            return None

    with env_extract.begin(write=False) as txn_read_extract:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_extract_single_from_processed, i, txn_read_extract, output_image_dir, 
                                custom_key_title, default_img_ext)
                for i in range(1, num_samples_extract + 1)
            ]
            for future in tqdm(as_completed(futures), total=num_samples_extract, desc='Extracting Images & Labels'):
                result = future.result()
                if result: extracted_label_entries.append(result)
    
    try:
        with open(output_gt_filepath, 'w', encoding='utf-8') as f_out_gt:
            for entry in extracted_label_entries:
                f_out_gt.write(entry + '\n')
    except IOError as e:
        print(f"Error writing final GT file {output_gt_filepath}: {e}"); env_extract.close(); return False

    if env_extract: env_extract.close()
    print(f'\nExtraction complete. {len(extracted_label_entries)} entries processed from {processed_lmdb_path}.')
    print(f"Extracted images saved to: {output_image_dir}")
    print(f"Final GT file saved to: {output_gt_filepath}")
    return True

# --- Main execution block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Reads an LMDB, creates a new LMDB ensuring a specific image name key exists, "
                    "and then extracts images and a GT file from the new LMDB."
    )
    
    parser.add_argument("--input_lmdb_path", type=str, required=True,
                        help="Path to the original LMDB dataset to process.")
    parser.add_argument("--new_lmdb_path", type=str, required=True,
                        help="Path where the new, processed LMDB (with guaranteed image name key) will be created.")
    parser.add_argument("--output_image_folder", type=str, required=True,
                        help="Folder where final images (extracted from the new LMDB) will be saved.")
    parser.add_argument("--output_gt_file", type=str, required=True,
                        help="Path where the final GT file (from new LMDB extraction) will be saved.")
    parser.add_argument("--custom_key_title", type=str, default="imgNameKey",
                        help="Base title for the custom key storing/retrieving image names (e.g., 'imgNameKey').")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() or 4,
                        help="Number of worker threads for extraction step.")
    parser.add_argument("--default_ext", type=str, default=".png",
                        help="Default image extension for derived names if key is missing or stored name is problematic.")

    args = parser.parse_args()

    # Step 1: Create the new LMDB with the guaranteed image name key
    new_lmdb_created = create_new_lmdb_with_guaranteed_name_key(
        input_lmdb_path=args.input_lmdb_path,
        output_new_lmdb_path=args.new_lmdb_path,
        custom_key_title=args.custom_key_title,
        default_img_ext=args.default_ext,
        num_workers=args.num_workers # num_workers for this step primarily affects reading speed if many small files
    )

    if not new_lmdb_created:
        print("Failed to create the new LMDB. Aborting extraction.")
    else:
        # Step 2: Extract images and GT file from the newly created LMDB
        extraction_successful = extract_images_and_gt_from_processed_lmdb(
            processed_lmdb_path=args.new_lmdb_path, # Use the new LMDB as input
            output_image_dir=args.output_image_folder,
            output_gt_filepath=args.output_gt_file,
            custom_key_title=args.custom_key_title,
            num_workers=args.num_workers,
            default_img_ext=args.default_ext
        )
        if extraction_successful:
            print(f"\nProcess complete. A new LMDB was created at '{args.new_lmdb_path}'.")
            print(f"You may now manually delete the old LMDB at '{args.input_lmdb_path}' if desired.")
        else:
            print("\nExtraction from the new LMDB failed.")
    
    print("\nScript finished.")

'''
Example Usage:

python 5_create_lmdb_with_imgname_key.py \
    --input_lmdb_path /ssd_scratch/cvit/lalitha/parseq/data/test/iiit-indic-hw-test/lmdb \
    --new_lmdb_path /ssd_scratch/cvit/lalitha/parseq/data/test/iiit-indic-hw-test/new_lmdb \
    --output_image_folder /ssd_scratch/cvit/lalitha/parseq/data/test/iiit-indic-hw-test/extracted_images_from_new_lmdb \
    --output_gt_file /ssd_scratch/cvit/lalitha/parseq/data/test/iiit-indic-hw-test/extracted_gt_from_new_lmdb.txt \
    --custom_key_title image_name 
'''