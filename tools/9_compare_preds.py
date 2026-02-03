import os
import argparse # Import the argparse library

def parse_data_file(filepath):
    """
    Reads a tab-separated file and parses its content into a dictionary.

    Args:
        filepath (str): The path to the input file.

    Returns:
        dict: A dictionary where keys are 'image_name' and values are
              another dictionary containing 'gt', 'pred', and 'correct'.
              Returns None if the file is not found.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        return None

    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        # Skip the header line
        next(f, None)
        for line in f:
            # Handle potential empty lines
            if not line.strip():
                continue
            
            try:
                image_name, gt, pred, correct_str = line.strip().split('\t')
                # Convert 'True'/'False' string to a boolean for easy comparison
                is_correct = correct_str.strip().lower() == 'true'
                data[image_name] = {
                    'gt': gt,
                    'pred': pred,
                    'correct': is_correct
                }
            except ValueError:
                print(f"Warning: Skipping malformed line in {filepath}: {line.strip()}")
                
    return data

def compare_and_filter_files(path_a, path_b, output_path):
    """
    Compares two result files and filters the data into three categories
    based on the correctness of their predictions.
    """
    print("--- Starting Comparison Process ---")
    
    # Step 1: Read and parse both files into dictionaries
    print(f"Reading data from File A: {path_a}")
    data_a = parse_data_file(path_a)
    if data_a is None:
        return

    print(f"Reading data from File B: {path_b}")
    data_b = parse_data_file(path_b)
    if data_b is None:
        return
    
    # Step 2: Initialize lists to hold the categorized results
    both_correct = []
    a_wrong_b_correct = []
    both_wrong = []
    
    print("\nComparing entries...")
    # Step 3: Iterate through common entries
    common_images = set(data_a.keys()) & set(data_b.keys())
    
    for image_name in common_images:
        entry_a = data_a[image_name]
        entry_b = data_b[image_name]
        
        if entry_a['gt'] != entry_b['gt']:
            print(f"Warning: Mismatched ground truth for image '{image_name}'. Skipping.")
            continue

        correct_a = entry_a['correct']
        correct_b = entry_b['correct']
        result_tuple = (image_name, entry_a['gt'], entry_a['pred'], entry_b['pred'])

        # Apply filtering logic
        if correct_a and correct_b:
            both_correct.append(result_tuple)
        elif not correct_a and correct_b:
            a_wrong_b_correct.append(result_tuple)
        elif not correct_a and not correct_b:
            both_wrong.append(result_tuple)
            
    print(f"Found {len(common_images)} common images to compare.")
    print(f" - Both Correct: {len(both_correct)}")
    print(f" - A Wrong, B Correct: {len(a_wrong_b_correct)}")
    print(f" - Both Wrong: {len(both_wrong)}")

    # Step 4: Write the categorized results to the output file
    print(f"\nWriting results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.write("="*20 + " 1. Both A and B are Correct " + "="*20 + "\n\n")
        f_out.write("image_name\tground_truth\tprediction_A\tprediction_B\n")
        for item in both_correct:
            f_out.write("\t".join(item) + "\n")
            
        f_out.write("\n\n" + "="*20 + " 2. A is Wrong, B is Correct " + "="*20 + "\n\n")
        f_out.write("image_name\tground_truth\tprediction_A\tprediction_B\n")
        for item in a_wrong_b_correct:
            f_out.write("\t".join(item) + "\n")
            
        f_out.write("\n\n" + "="*20 + " 3. Both A and B are Wrong " + "="*20 + "\n\n")
        f_out.write("image_name\tground_truth\tprediction_A\tprediction_B\n")
        for item in both_wrong:
            f_out.write("\t".join(item) + "\n")

    print("\n--- Process complete! ---")


if __name__ == "__main__":
    # --- This is the new section using argparse ---

    # 1. Create the parser
    parser = argparse.ArgumentParser(
        description="Compares two OCR result files and categorizes the differences based on correctness."
    )

    # 2. Add the arguments
    # These are "positional" arguments because they don't have a '-' or '--' prefix.
    # They are required by default.
    parser.add_argument("file_a_path", help="The full path to the first file (File A).")
    parser.add_argument("file_b_path", help="The full path to the second file (File B).")
    parser.add_argument("output_file_path", help="The full path for the output comparison file.")

    # 3. Parse the arguments from the command line
    args = parser.parse_args()

    # 4. Call the main function with the parsed arguments
    compare_and_filter_files(args.file_a_path, args.file_b_path, args.output_file_path)

'''
python 9_compare_preds.py \
    /home2/evanilalitha/10_parseq_code/generalized_hindi_HTR/icvgip_predictions/telugu/parseq/words.txt\
    /home2/evanilalitha/10_parseq_code/generalized_hindi_HTR/icvgip_predictions/telugu/pl_parseq/words.txt\
    /home2/evanilalitha/10_parseq_code/generalized_hindi_HTR/icvgip_predictions/telugu/fig_1.txt
python 9_compare_preds.py \
    /home2/evanilalitha/10_parseq_code/generalized_hindi_HTR/icvgip_predictions/telugu/parseq/uc.txt\
    /home2/evanilalitha/10_parseq_code/generalized_hindi_HTR/icvgip_predictions/telugu/pl_parseq/uc.txt\
    /home2/evanilalitha/10_parseq_code/generalized_hindi_HTR/icvgip_predictions/telugu/fig_2.txt
python 9_compare_preds.py \
    /home2/evanilalitha/10_parseq_code/generalized_hindi_HTR/icvgip_predictions/tamil/pl_parseq/uc.txt\
    /home2/evanilalitha/10_parseq_code/generalized_hindi_HTR/icvgip_predictions/tamil/cs_pl_parseq/uc.txt\
    /home2/evanilalitha/10_parseq_code/generalized_hindi_HTR/icvgip_predictions/tamil/fig_3A.txt
'''