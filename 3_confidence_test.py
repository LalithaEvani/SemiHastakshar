import argparse
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args
import numpy as np

def write_predictions_by_confidence(f, img_names, confidence_scores, predictions, confidence_ranges, moving_count, total_count, counts, labels=None):
    """Writes predictions grouped by confidence ranges and updates count for histogram."""
    moving_count = 0
    for idx, (lower, upper) in enumerate(confidence_ranges):
        count = sum(1 for c in confidence_scores if lower < c <= upper)
        counts[idx] += count  # Update the count for histogram
        f.write(f'\n\nconfidence {upper}-{lower}\n\n')
        if labels != None:
            total_correct = 0
            for img_name, confidence, label, prediction in zip(img_names, confidence_scores, labels, predictions):
                if lower < confidence <= upper:
                    f.write(f"{img_name}\t{confidence:.4f}\tlabel:{label}\tprediction:{prediction}\tcorrect:{label==prediction}\n")
                    correct = int(label==prediction)
                    total_correct+=correct

            f.write(f'\n\ntotal correct = {total_correct}')
        else:
            for img_name, confidence, label in zip(img_names, confidence_scores, predictions):
                if lower < confidence <= upper:
                    f.write(f"{img_name}\t{confidence:.4f}\t{label}\n")
        moving_count += count
        f.write(f'\n\ncount = {count}')
        f.write(f'\nPercentage {(moving_count / total_count) * 100:.2f}')
    return moving_count

def save_histogram(confidence_scores, output_path, bins=10):
    """Generates and saves a histogram of confidence scores with percentages and bin labels."""
    plt.figure(figsize=(10, 5))

    # Compute histogram data
    counts, bin_edges, patches = plt.hist(confidence_scores, bins=bins, color='blue', alpha=0.7, edgecolor='black')

    total_samples = sum(counts)  # Total number of predictions

    # Convert counts to percentages
    percentages = [(count / total_samples) * 100 for count in counts]

    # Add percentage text on top of bars
    for patch, percentage in zip(patches, percentages):
        plt.text(patch.get_x() + patch.get_width() / 2, patch.get_height(),
                 f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')

    plt.xlabel("Confidence Score Ranges")
    plt.ylabel("Number of Predictions")
    plt.title("Confidence Score Distribution")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    histogram_path = output_path.replace(".txt", "_histogram.png")
    plt.savefig(histogram_path, bbox_inches="tight")
    print(f"Histogram saved to {histogram_path}")

import csv

def write_predictions_to_csv(output_path, img_names, confidence_scores, predictions, labels=None):
    """Writes predictions to a CSV file."""
    with open(output_path, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        
        # Write header
        if labels is not None:
            writer.writerow(["Image_Name", "Confidence", "Label", "Prediction", "Correct"])
        else:
            writer.writerow(["Image_Name", "Confidence", "Prediction"])
        
        # Write rows
        for i in range(len(img_names)):
            row = [img_names[i], f"{confidence_scores[i]:.4f}"]
            if labels is not None:
                row.extend([labels[i], predictions[i], labels[i] == predictions[i]])
            else:
                row.append(predictions[i])
            writer.writerow(row)

    print(f"Predictions saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', 
                            # default='/ssd_scratch/cvit/lalitha/2_pdf_scraping_data/hindi_ckpt/parseq.ckpt',
                            default='/ssd_scratch/cvit/lalitha/2_pdf_scraping_data/PL/cycle_0_outputs/parseq/2025-03-19_19-27-23/checkpoints/epoch=27-step=34385-val_accuracy=90.7460-val_NED=97.3023-val_loss=0.3883.ckpt', 
                            help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', 
                            default='/ssd_scratch/cvit/lalitha/2_pdf_scraping_data/labelled/')
                            # default = '/ssd_scratch/cvit/lalitha/2_pdf_scraping_data/writer_1_some_pdfs/word_segmentation')
    parser.add_argument('--test_folder',
                            default='lmdb_data/', 
                            help='Name of the test folder')
    parser.add_argument('--image_root', 
                            help='Folder path containing images')
    parser.add_argument('--output_path', 
                            # default='/ssd_scratch/cvit/lalitha/2_pdf_scraping_data/labelled/outputs/predictions_1-0.txt', 
                            default = '/ssd_scratch/cvit/lalitha/2_pdf_scraping_data/PL/cycle_0_outputs/test_100_A/100_sample_A_prediction.txt',
                            help='Folder for saving the results with name of the txt file ')
                            # /ssd_scratch/cvit/lalitha/2_pdf_scraping_data/writer_1_some_pdfs/word_segmentation/outputs/predictions_1-0,9.txt
    parser.add_argument('--labelled', 
                            default=True, 
                            help='Whether test data is labelled or not')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')

    batch_size = 164
    num_workers = 4
    rotation = 0
    device = 'cuda'

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(device)
    hp = model.hparams
    datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
                                     hp.charset_test, batch_size, num_workers, False, rotation=rotation)
    print(f'Charset test used: {hp.charset_test}')

    test_set = sorted(set([args.test_folder]))

    confidence_ranges = [(0.9, 1.0), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.5, 0.6), 
                         (0.4, 0.5), (0.3, 0.4), (0.2, 0.3), (0.1, 0.2), (0.0, 0.1)]

    # confidence_ranges = [
    #     (0.90, 0.91), (0.91, 0.92), (0.92, 0.93), (0.93, 0.94), (0.94, 0.95), 
    #     (0.95, 0.96), (0.96, 0.97), (0.97, 0.98), (0.98, 0.99), (0.99, 1.00)
    # ]

    
    
    counts = [0] * len(confidence_ranges)  # Store counts for histogram
    moving_count = 0
    confidence_scores_list = []
    predictions_list = []
    img_names_list = []
    labels_list = []
    with open(args.output_path, 'w') as f:
        f.write("Image_Name\tConfidence\tLabel\n")  # Header
        for name, dataloader in datamodule.test_dataloaders(test_set).items():
            total = 0
            correct = 0
            cer = 0
            confidence = 0
            for img_names, imgs, labels in tqdm(iter(dataloader), desc=name):
                confidence_scores, predictions = model.generalized_test((imgs.to(model.device), labels))
                confidence_scores_list.extend(confidence_scores)
                predictions_list.extend(predictions)
                img_names_list.extend(img_names)
                if args.labelled:
                    labels_list.extend(labels)
                    res_dict = model.test_step((imgs.to(model.device), labels), -1)
                    res = res_dict['output']
                    total += res.num_samples
                    correct += res.correct
                    cer += res.cer
                    confidence +=res.confidence
            
            # Overall statistics
            # f.write(f'confidence 0.99+\n\n')
            # count = 0
            # for img_name, confidence, prediction in zip(img_names_list, confidence_scores_list, predictions_list):
            #     if confidence > 0.99:
            #         f.write(f"{img_name}\t{confidence:.4f}\tprediction:{prediction}\n")
            #         count+=1
            # moving_count += count
            # total_count = len(confidence_scores_list)
            # f.write(f'\n\ncount = {count}')
            # f.write(f'\nPercentage {(moving_count / total_count) * 100:.2f}')

            # Writing predictions for confidence ranges and updating histogram counts
            total_count = len(confidence_scores_list)
            if args.labelled:
                # f.write(f'confidence 0.99+\n\n')
                # count = 0
                # total_correct = 0
                # for img_name, confidence, label, prediction in zip(img_names, confidence_scores, labels, predictions):
                #     if confidence > 0.99:
                #         f.write(f"{img_name}\t{confidence:.4f}\tlabel:{label}\tprediction:{prediction}\tcorrect:{label==prediction}\n")
                #         correct = int(label==prediction)
                #         total_correct+=correct
                #         count+=1
                # f.write(f'\n\ncount = {count}')

                # f.write(f'\n\ntotal correct = {total_correct}')
                # f.write(f'\nPercentage correct{(total_correct/count) * 100:.2f}')
                # for img_name, confidence, prediction in zip(img_names_list, confidence_scores_list, predictions_list):
                #     if confidence > 0.99:
                #         f.write(f"{img_name}\t{confidence:.4f}\tprediction:{prediction}\n")
                #         count+=1
                # moving_count += count
                # total_count = len(confidence_scores_list)
                # f.write(f'\n\ncount = {count}')
                # f.write(f'\nPercentage {(moving_count / total_count) * 100:.2f}')
                moving_count = write_predictions_by_confidence(f, img_names_list, confidence_scores_list, predictions_list, confidence_ranges, moving_count, total_count, counts, labels=labels_list)
                accuracy = 100 * correct / total
                cer_total = 100 * cer / total
                wer = 100 - accuracy
                confidence_avg = confidence / total
                f.write(f'\n\nAccuracy: {accuracy} \nwer: {wer} \ncer: {cer_total} \nconfidence: {confidence_avg}')
            else:
                moving_count = write_predictions_by_confidence(f, img_names_list, confidence_scores_list, predictions_list, confidence_ranges, moving_count, total_count, counts)
    # Save histogram after processing all data
    # filtered_confidence_scores = [score for score in confidence_scores_list if 0.9 <= score <= 1.0]
    # save_histogram(filtered_confidence_scores, args.output_path)
    write_predictions_to_csv(args.output_path.replace(".txt", ".csv"), img_names_list, confidence_scores_list, predictions_list, labels_list if args.labelled else None)
    save_histogram(confidence_scores_list, args.output_path)
    print(f"Predictions saved to {args.output_path}")

if __name__ == '__main__':
    main()
