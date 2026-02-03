import argparse
import torch
# import matplotlib.pyplot as plt # Keep if needed for debugging
from tqdm import tqdm
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args
import numpy as np
# import csv # Keep if needed
import shutil
import os

def save_high_conf_images(output_path, img_names, confidence_scores, predictions, image_root, confidence_threshold):
    """Saves images with confidence > threshold in the output_path/images/ folder"""
    new_output_path = os.path.join(output_path, 'images/')
    os.makedirs(new_output_path, exist_ok=True)
    gt_file_path = os.path.join(output_path, "gt.txt")
    count = 0
    with open(gt_file_path, "w", encoding="utf-8") as gt_file:
        # Use tqdm for progress tracking if the list is large
        for img_name, confidence, prediction in tqdm(zip(img_names, confidence_scores, predictions), total=len(img_names), desc="Filtering High Confidence"):
            # Ensure confidence is a number
            try:
                # Confidence score might be a tensor, get the scalar value
                if isinstance(confidence, torch.Tensor):
                    conf_float = confidence.item()
                else:
                    conf_float = float(confidence)
            except (ValueError, TypeError) as e:
                print(f"Warning: Invalid confidence value '{confidence}' (type: {type(confidence)}) for image {img_name}. Skipping. Error: {e}")
                continue

            if conf_float > confidence_threshold:
                # Important: img_name from dataloader might be full path or relative.
                # We need the filename part to join with image_root correctly.
                img_filename = os.path.basename(img_name) # Get filename just in case
                img_path_source = os.path.join(image_root, img_name) # Try original name first
                img_path_target = os.path.join(new_output_path, img_filename)

                if not os.path.exists(img_path_source):
                     # Maybe img_name was absolute? Or image_root needs adjustment?
                     # Let's try joining image_root and just the filename
                     img_path_source_alt = os.path.join(image_root, img_filename)
                     if os.path.exists(img_path_source_alt):
                         img_path_source = img_path_source_alt
                     else:
                        # Log if still not found
                        print(f"Warning: Source image not found for high confidence sample. Tried: '{img_path_source}' and '{img_path_source_alt}' (img_name: {img_name}, image_root: {image_root})")
                        continue # Skip if image file missing


                if os.path.exists(img_path_source):
                    try:
                        shutil.copy(img_path_source, img_path_target)
                        # Write relative path (filename) for gt.txt, as LMDB script expects this relative to inputPath
                        gt_file.write(f"{img_filename}\t{prediction}\n")
                        count += 1
                    except Exception as e:
                         print(f"Error copying {img_path_source} to {img_path_target}: {e}")
                # else case handled above

    print(f"Saved {count} images with confidence > {confidence_threshold} to {new_output_path}")
    print(f"Ground truth saved to {gt_file_path}")


# Keep change_gt_file if you need a version that saves *all* predictions,
# but the PL pipeline uses save_high_conf_images which creates the filtered gt.txt
# def change_gt_file(output_path, img_names, predictions):
#     gt_file_path = os.path.join(output_path, "gt_all_preds.txt") # Avoid overwriting filtered gt.txt
#     with open(gt_file_path, "w", encoding='utf-8') as gt_file:
#         for img_name, prediction in zip(img_names, predictions):
#              img_filename = os.path.basename(img_name)
#              gt_file.write(f"{img_filename}\t{prediction}\n")
#     print(f'Saved all predictions in {gt_file_path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
                        required=True,
                        help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', required=True,
                        help="Root directory containing the LMDB folder specified by --test_folder")
    parser.add_argument('--test_folder', required=True,
                         help='Name of the LMDB test folder inside data_root (containing unlabeled data)')
    parser.add_argument('--image_root', required=True,
                        help='Folder path containing the original image files corresponding to the LMDB data')
    parser.add_argument('--output_path', required=True,
                        help='Output directory to save high-confidence images/ and gt.txt')
    parser.add_argument('--confidence_threshold', type=float, default=0.99,
                         help='Confidence threshold for selecting pseudo-labels') # Added threshold arg
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loader workers')
    parser.add_argument('--device', default='cuda', help='Device for inference (e.g., cuda, cpu)')

    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')

    device = args.device

    # Ensure output path exists
    os.makedirs(args.output_path, exist_ok=True)

    print(f"Loading model from {args.checkpoint}...")
    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(device)
    hp = model.hparams
    print("Model loaded successfully.")

    # Assume charset_train and charset_test are appropriate from the checkpoint's hparams
    # The datamodule needs these to correctly process any potential (dummy) labels from the LMDB
    try:
        print(f"Initializing datamodule with data_root={args.data_root}, batch_size={args.batch_size}...")
        datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
                                         hp.charset_test, args.batch_size, args.num_workers, False)
        print("Datamodule initialized.")
    except Exception as e:
        print(f"Error initializing DataModule: {e}")
        print(f"Please check data_root ('{args.data_root}') and test_folder ('{args.test_folder}') arguments.")
        return

    print(f'Using Charset test from model: {hp.charset_test}')

    # The test_folder argument should be the name of the LMDB directory within data_root
    test_set = [args.test_folder] # Use the specific folder name
    print(f"Getting dataloader for test set: {test_set}...")
    dataloaders = datamodule.test_dataloaders(test_set)

    if not dataloaders:
         print(f"Error: No dataloader found for test set '{args.test_folder}' in data_root '{args.data_root}'.")
         print("Ensure the LMDB directory exists and is named correctly.")
         return
    print("Dataloader obtained.")

    confidence_scores_list = []
    predictions_list = []
    img_names_list = []
    # labels_list = [] # Not needed for unlabeled data inference if labels are dummy

    # There should only be one dataloader based on test_set = [args.test_folder]
    for name, dataloader in dataloaders.items():
        print(f"Processing dataset: {name}")
        # Check if dataloader is empty
        if not dataloader or len(dataloader) == 0: # Added check for None or zero length
             print(f"Warning: Dataloader for '{name}' is empty or invalid. Skipping.")
             continue

        # Use the original user-provided inference loop structure
        for batch in tqdm(iter(dataloader), desc=f"Inferring on {name}", total=len(dataloader)):
            # Ensure batch has the expected structure (img_names, imgs, labels)
            # Even for unlabeled data, the dataloader usually yields dummy labels.
            if not isinstance(batch, (list, tuple)) or len(batch) != 3:
                print(f"Error: Unexpected batch format. Expected (img_names, imgs, labels), got {type(batch)} with len {len(batch) if hasattr(batch, '__len__') else 'N/A'}. Skipping batch.")
                continue

            img_names, imgs, labels = batch

            # Check if batch contents are valid
            if not img_names or imgs is None or labels is None:
                 print("Error: Invalid data within batch (e.g., empty lists, None tensors). Skipping batch.")
                 continue


            try:
                # --- Use model.generalized_test as per original code ---
                # We pass the labels (even if they are dummies from unlabeled LMDB)
                # as the method likely expects this structure.
                confidence_scores, predictions = model.generalized_test((imgs.to(model.device), labels))
                # -------------------------------------------------------

                # Ensure the outputs have the expected length matching the input batch size
                batch_size_actual = len(img_names)
                if len(confidence_scores) != batch_size_actual or len(predictions) != batch_size_actual:
                     print(f"Warning: Mismatch between input batch size ({batch_size_actual}) and output sizes (conf: {len(confidence_scores)}, pred: {len(predictions)}). Skipping batch results.")
                     continue

                # Extend the lists
                confidence_scores_list.extend(confidence_scores)
                predictions_list.extend(predictions)
                img_names_list.extend(img_names)

            except AttributeError:
                 print("Error: The loaded model does not have the 'generalized_test' method.")
                 print("Please ensure the checkpoint corresponds to a model with this method, or adapt the script.")
                 return # Stop execution if method is missing
            except Exception as e:
                 print(f"Error during model.generalized_test or list extension: {e}")
                 # Decide whether to skip batch or stop
                 continue

    # Save images and create gt.txt based on the threshold
    save_high_conf_images(args.output_path, img_names_list, confidence_scores_list, predictions_list, args.image_root, args.confidence_threshold)

    # Optionally call change_gt_file if you want *all* predictions saved separately
    # change_gt_file(args.output_path, img_names_list, predictions_list)

    print(f"High confidence prediction generation finished. Output at {args.output_path}")

if __name__ == '__main__':
    main()