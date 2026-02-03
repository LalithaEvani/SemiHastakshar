import argparse
import sys
from dataclasses import dataclass
from typing import List
import yaml
import torch
import os
from tqdm import tqdm
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args
from nltk import edit_distance

# Dataclass for storing aggregated results
@dataclass
class Result:
    dataset: str
    num_samples: int
    accuracy: float
    ned: float
    confidence: float
    label_length: float
    wer : float
    cer : float

# Function to print a formatted table of results
def print_results_table(results: List[Result], file=None):
    """Prints a formatted table for a list of Result objects."""
    w = max(map(len, map(getattr, results, ['dataset'] * len(results))))
    w = max(w, len('Dataset'), len('Combined'))
    print('| {:<{w}} | # samples | Accuracy | 1 - NED | Confidence | Label Length |      WER |      CER |'.format('Dataset', w=w), file=file)
    print('|:{:-<{w}}:|----------:|---------:|--------:|-----------:|-------------:|---------:|---------:|'.format('----', w=w), file=file)
    for res in results:
        print(f'| {res.dataset:<{w}} | {res.num_samples:>9} | {res.accuracy:>8.2f} | {res.ned:>7.2f} '
              f'| {res.confidence:>10.2f} | {res.label_length:>12.2f} | {res.wer:>8.2f} | {res.cer:>8.2f}', file=file)

@torch.inference_mode()
def main():
    """
    Main script for evaluating a model, logging per-sample predictions, and printing summary statistics.

    Example command:
    python evaluate_and_log.py \
    /path/to/your/model.ckpt \
    --test_folder=lmdb_data \
    --data_root=/path/to/your/data/ \
    --output_file=predictions_log.txt
    """
    parser = argparse.ArgumentParser(description="Evaluate a scene text recognition model and log predictions.")
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', default='./data', help='Root directory of training and test data')
    parser.add_argument('--test_folder', required=True, help='Name of the test folder inside data_root containing the LMDB dataset')
    parser.add_argument('--output_file', default='predictions.txt', help='Path to save the output file with predictions, GT, and confidence')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--charset_test', help='Path to a YAML config file to override the test charset', default=None)
    parser.add_argument('--change_charset_test', action='store_true', help='Set to use a different charset for testing')

    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')

    # Load the model from checkpoint
    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    hp = model.hparams

    # Setup the dataset and dataloader
    if args.change_charset_test:
        with open(args.charset_test, 'r') as file:
            config = yaml.safe_load(file)
        charset_test = config.get('model', {}).get('charset_test', "")
        print(f'Using custom test charset: {charset_test}')
    else:
        charset_test = hp.charset_test
        print(f'Using model\'s default test charset: {charset_test}')
    
    datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
                                     charset_test, args.batch_size, args.num_workers, False)

    # Get the specific dataloader for the test set
    try:
        dataloader = datamodule.test_dataloaders([args.test_folder])[args.test_folder]
    except KeyError:
        print(f"Error: Test set '{args.test_folder}' not found. Available sets: {datamodule.test_sets}")
        sys.exit(1)

    # Initialize metric accumulators
    total_samples = 0
    total_correct = 0
    total_ned = 0.0
    total_cer = 0.0
    total_confidence = 0.0
    total_label_length = 0
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    # Open the output file for writing
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        # Write header for the output file
        f_out.write("image_name\tground_truth\tprediction\tconfidence\n")
        
        # Loop over batches in the dataloader
        for img_names, imgs, labels_gt in tqdm(iter(dataloader), desc=f'Evaluating {args.test_folder}'):
            # Move images to the correct device
            imgs = imgs.to(model.device)
            
            # Forward pass through the model to get logits
            logits = model.forward(imgs)
            
            # Get probabilities and decode predictions
            probs = logits.softmax(-1)
            # `tokenizer.decode` returns raw predictions and per-character probabilities
            preds_raw, confidence_per_token = model.tokenizer.decode(probs)

            # Process each sample in the batch
            for i in range(len(imgs)):
                img_name = img_names[i]
                gt = labels_gt[i]
                raw_pred = preds_raw[i]

                # Apply the character set adapter to get the final prediction string
                pred = model.charset_adapter(raw_pred)

                # Calculate the confidence score for the entire sequence by multiplying token probabilities
                confidence = confidence_per_token[i].prod().item()

                # Write the result to the output file
                f_out.write(f"{img_name}\t{gt}\t{pred}\t{confidence:.4f}\n")

                # --- Update aggregate metrics ---
                if pred == gt:
                    total_correct += 1
                
                # Normalized Edit Distance (NED)
                gt_len = len(gt)
                pred_len = len(pred)
                if max(pred_len, gt_len) > 0:
                    total_ned += edit_distance(pred, gt) / max(pred_len, gt_len)

                # Character Error Rate (CER)
                if gt_len > 0:
                    total_cer += edit_distance(pred, gt) / gt_len

                total_confidence += confidence
                total_label_length += len(pred)
                total_samples += 1

    # --- Calculate final performance metrics ---
    if total_samples == 0:
        print("No samples found in the dataset. Exiting.")
        return

    accuracy = 100 * total_correct / total_samples
    wer = 100 - accuracy
    mean_ned = 100 * (1 - total_ned / total_samples)
    cer_total = 100 * total_cer / total_samples
    mean_conf = 100 * total_confidence / total_samples
    mean_label_length = total_label_length / total_samples

    # Create a Result object for printing
    result = Result(
        dataset=args.test_folder,
        num_samples=total_samples,
        accuracy=accuracy,
        ned=mean_ned,
        confidence=mean_conf,
        label_length=mean_label_length,
        wer=wer,
        cer=cer_total
    )

    # Print the results to the console
    print(f"\nEvaluation complete. Predictions logged to '{args.output_file}'")
    print("\n--- Performance Summary ---")
    print_results_table([result], file=sys.stdout)
    print("\n")


if __name__ == '__main__':
    main()

'''
python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/hindi/parseq_ckpts/parseq_hindi.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/hindi/test_data/ \
    --test_folder cs/ \
    --output_file /ssd_scratch/cvit/lalitha/hindi/confidence_txts/parseq/cs.txt \
    --batch_size 64
'''