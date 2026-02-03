import argparse
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args
import numpy as np
import csv
import shutil
import os

def save_high_conf_images(output_path, img_names, confidence_scores, predictions, image_root):
    """Saves images with confidence > 0.99 in the outputs folder"""
    new_output_path = os.path.join(output_path, 'images/')
    os.makedirs(new_output_path, exist_ok=True)
    gt_file_path = os.path.join(output_path, "gt.txt")
    with open(gt_file_path, "w", encoding="utf-8") as gt_file:
        for img_name, confidence, prediction in zip(img_names, confidence_scores, predictions):
            if confidence > 0.99:
                img_path = os.path.join(image_root, img_name)
                if os.path.exists(img_path):
                    shutil.copy(img_path,new_output_path)
                    gt_file.write(f"{img_name}\t{prediction}\n")
    print(f"High-confidence images saved to {output_path}")

def change_gt_file(output_path, img_names, predictions):
    gt_file_path = os.path.join(output_path, "gt.txt")
    with open(gt_file_path, "w", encoding='utf-8') as gt_file:
        for img_name, prediction in zip(img_names, predictions):
            gt_file.write(f"{img_name}\t{prediction}\n")
    print(f'saved the predicitons in the gt.txt file')
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', 
                        default='/ssd_scratch/cvit/lalitha/2_pdf_scraping_data/hindi_ckpt/parseq.ckpt', 
                        help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', 
                        default='/ssd_scratch/cvit/lalitha/2_pdf_scraping_data/unlabelled_one/500_sampled_test_data/')
    parser.add_argument('--test_folder',
                        default='lmdb_data/', 
                        help='Name of the test folder')
    parser.add_argument('--image_root',
                        help='Folder path containing images',
                        default='/ssd_scratch/cvit/lalitha/2_pdf_scraping_data/unlabelled_one/train_data/')
    parser.add_argument('--output_path', 
                        default='/ssd_scratch/cvit/lalitha/2_pdf_scraping_data/unlabelled_one/500_sampled_test_data/',
                        help='Output file for predictions')
    parser.add_argument('--labelled', 
                        default=False, 
                        help='Whether test data is labelled or not')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')

    batch_size = 50
    num_workers = 7
    device = 'cuda'

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(device)
    hp = model.hparams
    datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
                                     hp.charset_test, batch_size, num_workers, False)
    print(f'Charset test used: {hp.charset_test}')

    test_set = sorted(set([args.test_folder]))

    confidence_scores_list = []
    predictions_list = []
    img_names_list = []
    labels_list = []
    
    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        for img_names, imgs, labels in tqdm(iter(dataloader), desc=name):
            confidence_scores, predictions = model.generalized_test((imgs.to(model.device), labels))
            confidence_scores_list.extend(confidence_scores)
            predictions_list.extend(predictions)
            img_names_list.extend(img_names)
            if args.labelled:
                labels_list.extend(labels)
    
    # save_high_conf_images(args.output_path, img_names_list, confidence_scores_list, predictions_list, args.image_root)
    change_gt_file(args.output_path, img_names_list, predictions_list)
    print(f"Predictions saved to {args.output_path}")

if __name__ == '__main__':
    main()
