import os
import argparse

def parse_data_file(filepath):
    """
    Reads a tab-separated file and parses its content into a dictionary.
    This function is reusable and identical to the previous script.

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
        next(f, None)  # Skip header
        for line in f:
            if not line.strip():
                continue
            try:
                image_name, gt, pred, correct_str = line.strip().split('\t')
                is_correct = correct_str.strip().lower() == 'true'
                data[image_name] = {
                    'gt': gt,
                    'pred': pred,
                    'correct': is_correct
                }
            except ValueError:
                print(f"Warning: Skipping malformed line in {filepath}: {line.strip()}")
    return data

def find_ab_wrong_c_correct(path_a, path_b, path_c, output_path):
    """
    Compares three result files and filters for cases where A and B are
    wrong, but C is correct.
    """
    print("--- Starting Comparison Process ---")

    # Step 1: Read and parse all three files
    print(f"Reading data from File A: {path_a}")
    data_a = parse_data_file(path_a)
    if not data_a: return

    print(f"Reading data from File B: {path_b}")
    data_b = parse_data_file(path_b)
    if not data_b: return

    print(f"Reading data from File C: {path_c}")
    data_c = parse_data_file(path_c)
    if not data_c: return

    # Step 2: Find common images across all three files
    common_images = set(data_a.keys()) & set(data_b.keys()) & set(data_c.keys())
    if not common_images:
        print("Error: No common image_names found across the three files. Exiting.")
        return

    print(f"\nFound {len(common_images)} common images to compare across all three files.")
    
    # Step 3: Iterate through common images and apply filtering logic
    filtered_results = []
    for image_name in common_images:
        entry_a = data_a[image_name]
        entry_b = data_b[image_name]
        entry_c = data_c[image_name]

        # Data integrity check: ensure ground truth is consistent
        if not (entry_a['gt'] == entry_b['gt'] == entry_c['gt']):
            print(f"Warning: Mismatched ground truth for image '{image_name}'. Skipping.")
            continue

        # The core filtering condition
        if not entry_a['correct'] and not entry_b['correct'] and entry_c['correct']:
            # Store the relevant data for an informative output
            result_tuple = (
                image_name,
                entry_a['gt'],         # The ground truth
                entry_a['pred'],       # Prediction from A (Wrong)
                entry_b['pred'],       # Prediction from B (Wrong)
                entry_c['pred']        # Prediction from C (Correct)
            )
            filtered_results.append(result_tuple)
            
    print(f"Found {len(filtered_results)} entries where A & B were wrong, but C was correct.")

    # Step 4: Write the filtered results to the output file
    print(f"\nWriting results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        # Write a descriptive header for the file
        f_out.write("--- Cases where Models A & B Failed, but Model C Succeeded ---\n\n")
        
        # Write the column titles
        f_out.write("image_name\tground_truth\tprediction_A_wrong\tprediction_B_wrong\tprediction_C_correct\n")
        
        # Write the data
        for item in filtered_results:
            f_out.write("\t".join(item) + "\n")

    print("\n--- Process complete! ---")


if __name__ == "__main__":
    # Setup argparse to handle command-line arguments
    parser = argparse.ArgumentParser(
        description="Finds cases where models A and B fail, but model C succeeds. "
                    "All three input files must have the same format and common image names."
    )

    # Define the four required positional arguments
    parser.add_argument("file_a_path", help="Path to the result file for model A.")
    parser.add_argument("file_b_path", help="Path to the result file for model B.")
    parser.add_argument("file_c_path", help="Path to the result file for model C.")
    parser.add_argument("output_file_path", help="Path for the output file to be created.")

    # Parse the arguments provided by the user
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    find_ab_wrong_c_correct(
        args.file_a_path,
        args.file_b_path,
        args.file_c_path,
        args.output_file_path
    )


'''
python 10_compare_preds_ABC.py\
    /home2/evanilalitha/10_parseq_code/generalized_hindi_HTR/icvgip_predictions/tamil/parseq/uc.txt\
    /home2/evanilalitha/10_parseq_code/generalized_hindi_HTR/icvgip_predictions/tamil/pl_parseq/uc.txt\
    /home2/evanilalitha/10_parseq_code/generalized_hindi_HTR/icvgip_predictions/tamil/cs_pl_parseq/uc.txt\
    /home2/evanilalitha/10_parseq_code/generalized_hindi_HTR/icvgip_predictions/tamil/fig_3.txt
'''