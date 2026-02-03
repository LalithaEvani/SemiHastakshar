import argparse
import string
import sys
from dataclasses import dataclass
from typing import List
import lmdb
import yaml
import torch
import os
from tqdm import tqdm
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args
from nltk import edit_distance
from torch import Tensor

# This class should match the one in system.py. It represents the output for a single BATCH.
@dataclass
class BatchResult:
    num_samples: int
    correct: int
    ned: float
    cer: float
    confidence: float
    label_length: int
    loss: Tensor
    loss_numel: int
    preds: List[str]
    gts: List[str]
    confidences: List[float]

# This class is for storing the FINAL aggregated results for the summary table.
@dataclass
class Result:
    dataset: str
    num_samples: int
    accuracy: float
    ned: float
    confidence: float
    label_length: float
    wer: float
    cer: float

def print_results_table(results: List[Result], file=None):
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
    Example command:
    python test.py --checkpoint=/path/to/checkpoint.ckpt \
                   --test_folder=iiit-indic-hw-words/ \
                   --output_path=/path/to/predictions/file.txt
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', 
                            default='/ssd_scratch/cvit/lalitha/2_pdf_scraping_data/PL/pl_cycles_output/cycle_4/training_output_parseq/checkpoints/best.ckpt',
                            help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root',
                             default='/ssd_scratch/cvit/lalitha/data/test')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--cased', action='store_true', default=False, help='Cased comparison')
    parser.add_argument('--punctuation', action='store_true', default=False, help='Check punctuation')
    parser.add_argument('--new', action='store_true', default=False, help='Evaluate on new benchmark datasets')
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--test_folder', 
                            default='test/',
                            help='name of the test folder' )
    parser.add_argument('--charset_test', help='path to the YAML config charset file', default=None)
    parser.add_argument('--change_charset_test', help='want a different charset from that mentioned', default=False)
    parser.add_argument('--output_path', default='predictions.txt', help='Path to save the ground truth and predictions.')

    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    hp = model.hparams
    if args.change_charset_test:
        with open(args.charset_test, 'r') as file:
            config = yaml.safe_load(file)
        charset_test = config.get('model', {}).get('charset_test', "")
        datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
                                            charset_test, args.batch_size, args.num_workers, False, rotation=args.rotation)
        print(f'charset test used: {charset_test}')
    else:
        datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
                                         hp.charset_test, args.batch_size, args.num_workers, False, rotation=args.rotation)
        print(f'charset test used: {hp.charset_test}')
    
    test_set = sorted(set([args.test_folder]))
    results = {}
    max_width = max(map(len, test_set))

    # --- FIX: Create output directory if it doesn't exist ---
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    # --------------------------------------------------------

    with open(args.output_path, 'w', encoding='utf-8') as f_out:
        f_out.write("image_name\tground_truth\tprediction\tcorrect\n")

        for name, dataloader in datamodule.test_dataloaders(test_set).items():
            total = 0
            correct_total = 0 # Use a different name to avoid confusion
            ned = 0
            cer = 0
            confidence = 0
            label_length = 0

            for img_names, imgs, labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
                res_dict = model.test_step((img_names, imgs.to(model.device), labels), -1)
                res = res_dict['output'] # res is a BatchResult object

                # Accumulate stats for the summary table
                total += res.num_samples
                correct_total += res.correct
                ned += res.ned
                cer += res.cer
                confidence += res.confidence
                label_length += res.label_length

                # --- FIX: Correctly loop and write prediction results ---
                for i in range(res.num_samples):
                    img_name = img_names[i]
                    gt = res.gts[i]
                    pred = res.preds[i]
                    is_correct = (gt == pred)
                    f_out.write(f"{img_name}\t{gt}\t{pred}\t{is_correct}\n")
            
            # Calculate final metrics for the table
            accuracy = 100 * correct_total / total if total > 0 else 0
            cer_total = 100 * cer / total if total > 0 else 0
            mean_ned = 100 * (1 - ned / total) if total > 0 else 0
            mean_conf = 100 * confidence / total if total > 0 else 0
            mean_label_length = label_length / total if total > 0 else 0
            wer = 100 - accuracy
            
            # Use the correct Result class for the summary
            results[name] = Result(name, total, accuracy, mean_ned, mean_conf, mean_label_length, wer, cer_total)

    result_groups = {
        'Benchmark (Subset)': [args.test_folder]
    }
    if args.new:
        result_groups.update({'New': SceneTextDataModule.TEST_NEW})
    
    with open(args.checkpoint + '_all_writers_testset.log.txt', 'w') as f:
        for out in [f, sys.stdout]:
            for group, subset in result_groups.items():
                print(f'{group} set:', file=out)
                print_results_table([results[s] for s in subset if s in results], out)
                print('\n', file=out)

if __name__ == '__main__':
    main()
'''
python 8_test_pred_save.py --checkpoint=/ssd_scratch/cvit/lalitha/parseq_ckpt/2_pl_parseq/3_cs_internet_pl_parseq_with_paper_ckpts/tamil.ckpt \
      --data_root=/ssd_scratch/cvit/lalitha/data/tamil/uc \
      --output_path=/home2/evanilalitha/10_parseq_code/generalized_hindi_HTR/icvgip_predictions/tamil/cs_pl_parseq/uc.txt
'''