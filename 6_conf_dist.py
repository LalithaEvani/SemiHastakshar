import os
import lmdb
from PIL import Image
import io
from tqdm import tqdm  # For a nice progress bar

# --- Configuration ---

# 1. Set the paths to your input data
LMDB_PATH = '/ssd_scratch/cvit/lalitha/lmdb_data/internet_lmdb'
CONFIDENCE_FILE_PATH = '/ssd_scratch/cvit/lalitha/confidence_txts/2_telugu/internet_data.txt'

# 2. Set the base directory for the output folders
OUTPUT_BASE_DIR = '/ssd_scratch/cvit/lalitha/conf_dist_telugu/internet'

# 3. Set the number of intervals
NUM_INTERVALS = 10

# 4. Key prefixes from your LMDB creation script
LMDB_IMAGE_DATA_KEY_PREFIX = 'image'
LMDB_FILENAME_KEY_PREFIX = 'image_name'

# --- End of Configuration ---


def process_data():
    """
    Reads a confidence file, looks up the corresponding image data and filename
    in the LMDB, ensures the filename has an extension, and sorts into folders.
    """
    print("Starting the process with the direct lookup method...")
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    print(f"LMDB Path: {LMDB_PATH}")
    print(f"Confidence File Path: {CONFIDENCE_FILE_PATH}")
    print(f"Output Path: {OUTPUT_BASE_DIR}")

    try:
        env = lmdb.open(LMDB_PATH, readonly=True, lock=False, readahead=False, meminit=False)
    except lmdb.Error as e:
        print(f"FATAL ERROR: Could not open LMDB database at '{LMDB_PATH}'. Error: {e}")
        return

    try:
        with open(CONFIDENCE_FILE_PATH, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
    except FileNotFoundError:
        print(f"FATAL ERROR: Confidence file not found at '{CONFIDENCE_FILE_PATH}'.")
        env.close()
        return

    with env.begin(write=False) as txn:
        total_lines = len(lines)
        print(f"\nProcessing {total_lines} entries from the confidence file...")
        
        processed_count = 0
        skipped_count = 0

        for line in tqdm(lines, desc="Sorting images by confidence"):
            line = line.strip()
            if not line:
                continue

            try:
                parts = line.split('\t')
                index_from_txt = parts[0]
                index_from_txt = os.path.splitext(index_from_txt)[0]
                confidence = float(parts[3])
            except (ValueError, IndexError):
                skipped_count += 1
                continue

            image_data_key = f"{LMDB_IMAGE_DATA_KEY_PREFIX}-{index_from_txt}".encode()
            filename_key = f"{LMDB_FILENAME_KEY_PREFIX}-{index_from_txt}".encode()
            # print(f'filename_key: {filename_key}, image_data_key: {image_data_key}')

            image_buf = txn.get(image_data_key)
            filename_bytes = txn.get(filename_key)

            if image_buf is None:
                skipped_count += 1
                continue

            # ======================================================================
            # === FIX IS HERE: Determine filename and ensure it has an extension ===
            # ======================================================================
            base_filename = ""
            if filename_bytes:
                try:
                    base_filename = filename_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    base_filename = index_from_txt
            else:
                base_filename = index_from_txt
            
            # Check if the filename from the LMDB already has an extension.
            # If not, add a default '.png' extension. This prevents the error.
            name_part, ext_part = os.path.splitext(base_filename)
            if not ext_part:
                output_filename = f"{name_part}.png"
            else:
                output_filename = base_filename
            # ======================================================================
            # === END OF FIX                                                     ===
            # ======================================================================

            interval_index = NUM_INTERVALS - 1 if confidence == 1.0 else int(confidence * NUM_INTERVALS)
            lower_bound = interval_index / NUM_INTERVALS
            upper_bound = (interval_index + 1) / NUM_INTERVALS
            interval_folder_name = f"confidence_{lower_bound:.1f}_to_{upper_bound:.1f}"

            interval_dir = os.path.join(OUTPUT_BASE_DIR, interval_folder_name)
            output_images_dir = os.path.join(interval_dir, 'images')
            output_txt_path = os.path.join(interval_dir, 'confidence_scores.txt')

            os.makedirs(output_images_dir, exist_ok=True)
            with open(output_txt_path, 'a', encoding='utf-8') as f_out:
                f_out.write(line + '\n')

            try:
                image = Image.open(io.BytesIO(image_buf))
                image.save(os.path.join(output_images_dir, output_filename))
                processed_count += 1
            except Exception as e:
                print(f"Error saving image '{output_filename}' for index '{index_from_txt}': {e}")
                skipped_count += 1

    env.close()
    print("\n--- Processing Complete ---")
    print(f"Total entries processed and saved: {processed_count}")
    print(f"Total entries skipped (not found or error): {skipped_count}")
    print(f"Check the '{OUTPUT_BASE_DIR}' directory for results.")


if __name__ == "__main__":
    process_data()