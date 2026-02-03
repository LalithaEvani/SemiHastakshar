#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Processes text files containing confidence scores to generate and save distribution plots.

This script walks a specified input directory recursively, finds all '.txt' files,
and for each file, it:
1.  Reads the data, which is expected to be in a tab-separated format with at
    least four columns: [image_index, ground_truth, prediction, confidence].
2.  Parses the data robustly, handling potential malformed lines.
3.  Calculates the distribution of confidence scores across 10 bins (0.0-0.1, ..., 0.9-1.0).
4.  Generates a bar plot showing the distribution, with percentage labels on each bar.
5.  Saves the plot as a PNG image in a corresponding output directory, preserving the
    original folder structure.

The input and output directories are specified via command-line arguments.

Example Usage:
    python your_script_name.py --data_root /path/to/confidence_txts --output_root /path/for/output_plots
    
    # Using short aliases:
    python your_script_name.py -d /path/to/confidence_txts -o /path/for/output_plots
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --- The Plotting Function (with robust data loading) ---

def generate_and_save_distribution_plot(input_file_path, output_image_path):
    """
    Reads a confidence score file with a robust parser, generates a
    distribution plot with percentages, and saves it.
    
    Args:
        input_file_path (str): The full path to the input .txt file.
        output_image_path (str): The full path where the output .png plot will be saved.
    """
    try:
        # Use a robust data loading method to prevent crashes from malformed lines
        df = pd.read_csv(
            input_file_path,
            sep='\t',
            header=None,
            names=['image_index', 'ground_truth', 'prediction', 'confidence'],
            on_bad_lines='skip',
            engine='python',
            dtype=str
        )

        if df.empty:
            return

        # Clean the 'confidence' column
        df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
        df.dropna(subset=['confidence'], inplace=True)
        
        total_valid_images = len(df)
        if total_valid_images == 0:
            return

        # Create confidence bins and calculate distribution
        bins = np.linspace(0, 1, 11)
        df['confidence_bin'] = pd.cut(df['confidence'], bins=bins, include_lowest=True, right=True)
        distribution = df['confidence_bin'].value_counts().sort_index()

        # --- Plotting Logic ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(14, 8))
        
        bar_labels = [str(interval) for interval in distribution.index]
        bars = ax.bar(
            bar_labels,
            distribution.values,
            color='darkcyan',
            edgecolor='black'
        )

        # Add percentage annotations
        for bar in bars:
            height = bar.get_height()
            percentage = (height / total_valid_images) * 100
            ax.annotate(
                f'{percentage:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=20, fontweight='bold'
            )

        file_title = os.path.basename(input_file_path)
        ax.set_title(f'Confidence Distribution for: {file_title}\n(Total Valid Images: {total_valid_images})', fontsize=16, fontweight='bold')
        ax.set_xlabel('Confidence Interval', fontsize=12)
        ax.set_ylabel('Number of Images', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(output_image_path, dpi=100)
        plt.close(fig)

    except Exception as e:
        print(f"  - UNEXPECTED ERROR while processing file '{input_file_path}'. Reason: {e}")


# --- Main Processing Logic ---

def main(data_root, output_root):
    """
    Walks the input directory, finds all .txt files, and processes them.

    Args:
        data_root (str): The root directory containing input .txt files.
        output_root (str): The root directory where output plots will be saved.
    """
    print(f"Starting process...")
    print(f"Input Data Root:  {data_root}")
    print(f"Output Root:      {output_root}")

    # Find all .txt files to process
    files_to_process = []
    for root, _, files in os.walk(data_root):
        for file in files:
            if file.endswith('.txt'):
                files_to_process.append(os.path.join(root, file))

    if not files_to_process:
        print("No .txt files found in the input directory. Exiting.")
        return

    print(f"\nFound {len(files_to_process)} text files to process.")

    # Process each file with a progress bar
    for input_path in tqdm(files_to_process, desc="Generating Distributions"):
        # Create the corresponding output directory structure
        relative_path = os.path.relpath(input_path, data_root)
        output_sub_dir = os.path.dirname(os.path.join(output_root, relative_path))
        os.makedirs(output_sub_dir, exist_ok=True)
        
        # Define the output image path
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_image_path = os.path.join(output_sub_dir, f"{base_name}.png")
        
        # Generate and save the plot
        generate_and_save_distribution_plot(input_path, output_image_path)

    print("\nProcessing complete!")
    print(f"All percentage distribution plots have been saved in '{output_root}'")

# --- Command-Line Argument Parsing and Script Entry Point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate confidence distribution plots from text files. The script maintains the original directory structure from the input in the output location.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # =======================================================================
    # === CHANGE IS HERE: Using named arguments instead of positional ones  ===
    # =======================================================================
    parser.add_argument(
        "-d", "--data_root",
        type=str,
        required=True,
        help="The root directory containing the input .txt confidence files."
    )
    parser.add_argument(
        "-o", "--output_root",
        type=str,
        required=True,
        help="The root directory where the output .png plots will be saved."
    )
    # =======================================================================
    # === END OF CHANGE                                                   ===
    # =======================================================================
    
    args = parser.parse_args()
    
    # Call the main function with the provided arguments
    main(args.data_root, args.output_root)

"""
python 7_conf_dist_graphs_all.py \
    --data_root /ssd_scratch/cvit/lalitha/hindi/confidence_txts/parseq \
    --output_root /ssd_scratch/cvit/lalitha/hindi/confidence_graphs/parseq
"""