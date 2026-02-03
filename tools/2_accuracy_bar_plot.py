import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_confidence_distribution(csv_path, output_path):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Define confidence bins from 0 to 1 with a step of 0.1
    bins = np.arange(0, 1.1, 0.1)
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    
    # Assign confidence values to bins
    df['Confidence Bin'] = pd.cut(df['Confidence'], bins=bins, labels=labels, include_lowest=True)
    
    # Count total samples and correct predictions per bin
    total_counts = df['Confidence Bin'].value_counts().sort_index()
    correct_counts = df[df['Correct'] == True]['Confidence Bin'].value_counts().sort_index()
    
    # Ensure all bins are present in both series
    total_counts = total_counts.reindex(labels, fill_value=0)
    correct_counts = correct_counts.reindex(labels, fill_value=0)
    
    # Plot the bar graph
    x = np.arange(len(labels))  # Label positions
    width = 0.4  # Bar width
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, total_counts, width, label='Confidence score')
    ax.bar(x + width/2, correct_counts, width, label='Correct Predictions')
    
    # Formatting the plot
    ax.set_xlabel('Confidence Interval')
    ax.set_ylabel('Count')
    ax.set_title('Confidence and Accuracy Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()
    ax.grid(True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    csv_path = "/ssd_scratch/cvit/lalitha/2_pdf_scraping_data/PL/outputs/predictions_1-0.csv"
    output_path = "/ssd_scratch/cvit/lalitha/2_pdf_scraping_data/PL/outputs/accuracy_conf_plt.png"
    
    plot_confidence_distribution(csv_path, output_path)
