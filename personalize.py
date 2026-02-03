import argparse
import os
import torch
from tqdm import tqdm
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--test_folder', help='Name of the test folder')
    parser.add_argument('--image_root', help='Folder path containing images')
    parser.add_argument('--output_path', default='output/predictions.txt', help='Folder for saving the results with name of the txt file ')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')

    batch_size = 512
    num_workers = 4
    rotation = 0
    device = 'cuda'

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(device)
    hp = model.hparams
    datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
                                     hp.charset_test, batch_size, num_workers, False, rotation=rotation)
    print(f'Charset test used: {hp.charset_test}')

    max_width = max(map(len, args.test_folder))
    test_set = sorted(set([args.test_folder]))
    
    # os.makedirs(args.output_path, exist_ok=True)
    # output_file_path = os.path.join(args.output_path, 'predictions.txt')
    moving_count = 0
    count = 0
    total_count = 0
    with open(args.output_path, 'w') as f:
        f.write("Image_Name\tConfidence\tLabel\n")  # Header
        for name, dataloader in datamodule.test_dataloaders(test_set).items():
            for img_names, imgs, labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
                confidence_scores, predictions = model.generalized_test((imgs.to(model.device), labels))
                f.write(f'confidence 0.99\n\n')
                for img_name, confidence, label in zip(img_names, confidence_scores, predictions):
                    if confidence > 0.99:
                        f.write(f"{img_name}\t{confidence:.4f}\t{label}\n")
                        count+=1
                    total_count+=1
                moving_count+=count
                f.write(f'\n\ncount = {count}')
                f.write(f'\nPercentage {(moving_count/total_count)*100}')
                count = 0 
                f.write(f'\n\nconfidence 0.99-0.98\n\n')
                for img_name, confidence, label in zip(img_names, confidence_scores, predictions):
                    if confidence < 0.99 and confidence >0.98:
                        f.write(f"{img_name}\t{confidence:.4f}\t{label}\n")
                        count+=1
                moving_count+=count
                f.write(f'\n\ncount = {count}')
                f.write(f'\nPercentage {(moving_count/total_count)*100}')


                count = 0 
                f.write(f'\n\nconfidence 0.98-0.97\n\n')
                for img_name, confidence, label in zip(img_names, confidence_scores, predictions):
                    if confidence < 0.98 and confidence >0.97:
                        f.write(f"{img_name}\t{confidence:.4f}\t{label}\n")
                        count+=1
                moving_count+=count
                f.write(f'\n\ncount = {count}')
                f.write(f'\nPercentage {(moving_count/total_count)*100}')

                count = 0 
                f.write(f'\n\nconfidence 0.97-0.96\n\n')
                for img_name, confidence, label in zip(img_names, confidence_scores, predictions):
                    if confidence < 0.97 and confidence >0.96:
                        f.write(f"{img_name}\t{confidence:.4f}\t{label}\n")
                        count+=1
                moving_count+=count
                f.write(f'\n\ncount = {count}')
                f.write(f'\nPercentage {(moving_count/total_count)*100}')

                count = 0 
                f.write(f'\n\nconfidence 0.96-0.95\n\n')
                for img_name, confidence, label in zip(img_names, confidence_scores, predictions):
                    if confidence < 0.96 and confidence >0.95:
                        f.write(f"{img_name}\t{confidence:.4f}\t{label}\n")
                        count+=1
                moving_count+=count
                f.write(f'\n\ncount = {count}')
                f.write(f'\nPercentage {(moving_count/total_count)*100}')


                count = 0 
                f.write(f'\n\nconfidence 0.95-0.94\n\n')
                for img_name, confidence, label in zip(img_names, confidence_scores, predictions):
                    if confidence < 0.95 and confidence >0.94:
                        f.write(f"{img_name}\t{confidence:.4f}\t{label}\n")
                        count+=1
                moving_count+=count
                f.write(f'\n\ncount = {count}')
                f.write(f'\nPercentage {(moving_count/total_count)*100}')

                count = 0 
                f.write(f'\n\nconfidence 0.93-0.92\n\n')
                for img_name, confidence, label in zip(img_names, confidence_scores, predictions):
                    if confidence < 0.93 and confidence >0.92:
                        f.write(f"{img_name}\t{confidence:.4f}\t{label}\n")
                        count+=1
                moving_count+=count
                f.write(f'\n\ncount = {count}')
                f.write(f'\nPercentage {(moving_count/total_count)*100}')

                count = 0 
                f.write(f'\n\nconfidence 0.92-0.91\n\n')
                for img_name, confidence, label in zip(img_names, confidence_scores, predictions):
                    if confidence < 0.92 and confidence >0.91:
                        f.write(f"{img_name}\t{confidence:.4f}\t{label}\n")
                        count+=1
                moving_count+=count
                f.write(f'\n\ncount = {count}')
                f.write(f'\nPercentage {(moving_count/total_count)*100}')

    print(f"Predictions saved to {args.output_path}")

if __name__ == '__main__':
    main()
