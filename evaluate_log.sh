#!/bin/bash
#SBATCH -A evanilalitha
#SBATCH --nodelist=gnode049
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1-00:00:00
#SBATCH --output=bash_outputs/evaluate_log_pl_parseq.txt
#SBATCH --mail-type=END

conda activate parseq
cd /home2/evanilalitha/10_generalized_hindi_HTR/generalized_hindi_HTR/

# hindi
python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/hindi/parseq_ckpts/parseq_hindi.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/hindi/test_data/ \
    --test_folder cs/ \
    --output_file /ssd_scratch/cvit/lalitha/hindi/confidence_txts/parseq/cs.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/hindi/parseq_ckpts/parseq_hindi.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/hindi/test_data/ \
    --test_folder iiit/ \
    --output_file /ssd_scratch/cvit/lalitha/hindi/confidence_txts/parseq/iiit.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/hindi/parseq_ckpts/parseq_hindi.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/hindi/test_data/ \
    --test_folder internet/ \
    --output_file /ssd_scratch/cvit/lalitha/hindi/confidence_txts/parseq/internet.txt \
    --batch_size 64

python 7_conf_dist_graphs_all.py \
    --data_root /ssd_scratch/cvit/lalitha/hindi/confidence_txts/parseq \
    --output_root /ssd_scratch/cvit/lalitha/hindi/confidence_graphs/parseq

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/hindi/parseq_ckpts/parseq_unlabelled_exp1.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/hindi/test_data/ \
    --test_folder cs/ \
    --output_file /ssd_scratch/cvit/lalitha/hindi/confidence_txts/pl_parseq_exp1/cs.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/hindi/parseq_ckpts/parseq_unlabelled_exp1.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/hindi/test_data/ \
    --test_folder iiit/ \
    --output_file /ssd_scratch/cvit/lalitha/hindi/confidence_txts/pl_parseq_exp1/iiit.txt \
    --batch_size 64

python 5_evaluate_log.py \
   /ssd_scratch/cvit/lalitha/hindi/parseq_ckpts/parseq_unlabelled_exp1.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/hindi/test_data/ \
    --test_folder internet/ \
    --output_file /ssd_scratch/cvit/lalitha/hindi/confidence_txts/pl_parseq_exp1/internet.txt \
    --batch_size 64

python 7_conf_dist_graphs_all.py \
    --data_root /ssd_scratch/cvit/lalitha/hindi/confidence_txts/pl_parseq_exp1 \
    --output_root /ssd_scratch/cvit/lalitha/hindi/confidence_graphs/pl_parseq_exp1

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/hindi/parseq_ckpts/parseq_unlabelled_exp2.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/hindi/test_data/ \
    --test_folder cs/ \
    --output_file /ssd_scratch/cvit/lalitha/hindi/confidence_txts/pl_parseq_exp2/cs.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/hindi/parseq_ckpts/parseq_unlabelled_exp2.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/hindi/test_data/ \
    --test_folder iiit/ \
    --output_file /ssd_scratch/cvit/lalitha/hindi/confidence_txts/pl_parseq_exp2/iiit.txt \
    --batch_size 64

python 5_evaluate_log.py \
   /ssd_scratch/cvit/lalitha/hindi/parseq_ckpts/parseq_unlabelled_exp2.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/hindi/test_data/ \
    --test_folder internet/ \
    --output_file /ssd_scratch/cvit/lalitha/hindi/confidence_txts/pl_parseq_exp2/internet.txt \
    --batch_size 64

python 7_conf_dist_graphs_all.py \
    --data_root /ssd_scratch/cvit/lalitha/hindi/confidence_txts/pl_parseq_exp2 \
    --output_root /ssd_scratch/cvit/lalitha/hindi/confidence_graphs/pl_parseq_exp2

# bengali

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/bengali/parseq_ckpts/parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/bengali/test_data/ \
    --test_folder cs/ \
    --output_file /ssd_scratch/cvit/lalitha/bengali/confidence_txts/parseq/cs.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/bengali/parseq_ckpts/parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/bengali/test_data/ \
    --test_folder iiit/ \
    --output_file /ssd_scratch/cvit/lalitha/bengali/confidence_txts/parseq/iiit.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/bengali/parseq_ckpts/parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/bengali/test_data/ \
    --test_folder internet/ \
    --output_file /ssd_scratch/cvit/lalitha/bengali/confidence_txts/parseq/internet.txt \
    --batch_size 64

python 7_conf_dist_graphs_all.py \
    --data_root /ssd_scratch/cvit/lalitha/bengali/confidence_txts/parseq \
    --output_root /ssd_scratch/cvit/lalitha/bengali/confidence_graphs/parseq

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/bengali/parseq_ckpts/pl_parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/bengali/test_data/ \
    --test_folder cs/ \
    --output_file /ssd_scratch/cvit/lalitha/bengali/confidence_txts/pl_parseq/cs.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/bengali/parseq_ckpts/pl_parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/bengali/test_data/ \
    --test_folder iiit/ \
    --output_file /ssd_scratch/cvit/lalitha/bengali/confidence_txts/pl_parseq/iiit.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/bengali/parseq_ckpts/pl_parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/bengali/test_data/ \
    --test_folder internet/ \
    --output_file /ssd_scratch/cvit/lalitha/bengali/confidence_txts/pl_parseq/internet.txt \
    --batch_size 64

python 7_conf_dist_graphs_all.py \
    --data_root /ssd_scratch/cvit/lalitha/bengali/confidence_txts/pl_parseq \
    --output_root /ssd_scratch/cvit/lalitha/bengali/confidence_graphs/pl_parseq

# gujarati

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/gujarati/parseq_ckpts/parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/gujarati/test_data/ \
    --test_folder cs/ \
    --output_file /ssd_scratch/cvit/lalitha/gujarati/confidence_txts/parseq/cs.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/gujarati/parseq_ckpts/parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/gujarati/test_data/ \
    --test_folder iiit/ \
    --output_file /ssd_scratch/cvit/lalitha/gujarati/confidence_txts/parseq/iiit.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/gujarati/parseq_ckpts/parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/gujarati/test_data/ \
    --test_folder internet/ \
    --output_file /ssd_scratch/cvit/lalitha/gujarati/confidence_txts/parseq/internet.txt \
    --batch_size 64

python 7_conf_dist_graphs_all.py \
    --data_root /ssd_scratch/cvit/lalitha/gujarati/confidence_txts/parseq \
    --output_root /ssd_scratch/cvit/lalitha/gujarati/confidence_graphs/parseq

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/gujarati/parseq_ckpts/pl_parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/gujarati/test_data/ \
    --test_folder cs/ \
    --output_file /ssd_scratch/cvit/lalitha/gujarati/confidence_txts/pl_parseq/cs.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/gujarati/parseq_ckpts/pl_parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/gujarati/test_data/ \
    --test_folder iiit/ \
    --output_file /ssd_scratch/cvit/lalitha/gujarati/confidence_txts/pl_parseq/iiit.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/gujarati/parseq_ckpts/pl_parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/gujarati/test_data/ \
    --test_folder internet/ \
    --output_file /ssd_scratch/cvit/lalitha/gujarati/confidence_txts/pl_parseq/internet.txt \
    --batch_size 64

python 7_conf_dist_graphs_all.py \
    --data_root /ssd_scratch/cvit/lalitha/gujarati/confidence_txts/pl_parseq \
    --output_root /ssd_scratch/cvit/lalitha/gujarati/confidence_graphs/pl_parseq

#kannada

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/kannada/parseq_ckpts/parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/kannada/test_data/ \
    --test_folder cs/ \
    --output_file /ssd_scratch/cvit/lalitha/kannada/confidence_txts/parseq/cs.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/kannada/parseq_ckpts/parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/kannada/test_data/ \
    --test_folder iiit/ \
    --output_file /ssd_scratch/cvit/lalitha/kannada/confidence_txts/parseq/iiit.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/kannada/parseq_ckpts/parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/kannada/test_data/ \
    --test_folder internet/ \
    --output_file /ssd_scratch/cvit/lalitha/kannada/confidence_txts/parseq/internet.txt \
    --batch_size 64

python 7_conf_dist_graphs_all.py \
    --data_root /ssd_scratch/cvit/lalitha/kannada/confidence_txts/parseq \
    --output_root /ssd_scratch/cvit/lalitha/kannada/confidence_graphs/parseq

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/kannada/parseq_ckpts/pl_parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/kannada/test_data/ \
    --test_folder cs/ \
    --output_file /ssd_scratch/cvit/lalitha/kannada/confidence_txts/pl_parseq/cs.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/kannada/parseq_ckpts/pl_parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/kannada/test_data/ \
    --test_folder iiit/ \
    --output_file /ssd_scratch/cvit/lalitha/kannada/confidence_txts/pl_parseq/iiit.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/kannada/parseq_ckpts/pl_parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/kannada/test_data/ \
    --test_folder internet/ \
    --output_file /ssd_scratch/cvit/lalitha/kannada/confidence_txts/pl_parseq/internet.txt \
    --batch_size 64

python 7_conf_dist_graphs_all.py \
    --data_root /ssd_scratch/cvit/lalitha/kannada/confidence_txts/pl_parseq \
    --output_root /ssd_scratch/cvit/lalitha/kannada/confidence_graphs/pl_parseq

# malayalam

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/malayalam/parseq_ckpts/parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/malayalam/test_data/ \
    --test_folder cs/ \
    --output_file /ssd_scratch/cvit/lalitha/malayalam/confidence_txts/parseq/cs.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/malayalam/parseq_ckpts/parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/malayalam/test_data/ \
    --test_folder iiit/ \
    --output_file /ssd_scratch/cvit/lalitha/malayalam/confidence_txts/parseq/iiit.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/malayalam/parseq_ckpts/parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/malayalam/test_data/ \
    --test_folder internet/ \
    --output_file /ssd_scratch/cvit/lalitha/malayalam/confidence_txts/parseq/internet.txt \
    --batch_size 64

python 7_conf_dist_graphs_all.py \
    --data_root /ssd_scratch/cvit/lalitha/malayalam/confidence_txts/parseq \
    --output_root /ssd_scratch/cvit/lalitha/malayalam/confidence_graphs/parseq

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/malayalam/parseq_ckpts/pl_parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/malayalam/test_data/ \
    --test_folder cs/ \
    --output_file /ssd_scratch/cvit/lalitha/malayalam/confidence_txts/pl_parseq/cs.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/malayalam/parseq_ckpts/pl_parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/malayalam/test_data/ \
    --test_folder iiit/ \
    --output_file /ssd_scratch/cvit/lalitha/malayalam/confidence_txts/pl_parseq/iiit.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/malayalam/parseq_ckpts/pl_parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/malayalam/test_data/ \
    --test_folder internet/ \
    --output_file /ssd_scratch/cvit/lalitha/malayalam/confidence_txts/pl_parseq/internet.txt \
    --batch_size 64

python 7_conf_dist_graphs_all.py \
    --data_root /ssd_scratch/cvit/lalitha/malayalam/confidence_txts/pl_parseq \
    --output_root /ssd_scratch/cvit/lalitha/malayalam/confidence_graphs/pl_parseq

# tamil
python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/tamil/parseq_ckpts/parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/tamil/test_data/ \
    --test_folder cs/ \
    --output_file /ssd_scratch/cvit/lalitha/tamil/confidence_txts/parseq/cs.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/tamil/parseq_ckpts/parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/tamil/test_data/ \
    --test_folder iiit/ \
    --output_file /ssd_scratch/cvit/lalitha/tamil/confidence_txts/parseq/iiit.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/tamil/parseq_ckpts/parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/tamil/test_data/ \
    --test_folder internet/ \
    --output_file /ssd_scratch/cvit/lalitha/tamil/confidence_txts/parseq/internet.txt \
    --batch_size 64

python 7_conf_dist_graphs_all.py \
    --data_root /ssd_scratch/cvit/lalitha/tamil/confidence_txts/parseq \
    --output_root /ssd_scratch/cvit/lalitha/tamil/confidence_graphs/parseq

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/tamil/parseq_ckpts/pl_parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/tamil/test_data/ \
    --test_folder cs/ \
    --output_file /ssd_scratch/cvit/lalitha/tamil/confidence_txts/pl_parseq/cs.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/tamil/parseq_ckpts/pl_parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/tamil/test_data/ \
    --test_folder iiit/ \
    --output_file /ssd_scratch/cvit/lalitha/tamil/confidence_txts/pl_parseq/iiit.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/tamil/parseq_ckpts/pl_parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/tamil/test_data/ \
    --test_folder internet/ \
    --output_file /ssd_scratch/cvit/lalitha/tamil/confidence_txts/pl_parseq/internet.txt \
    --batch_size 64

python 7_conf_dist_graphs_all.py \
    --data_root /ssd_scratch/cvit/lalitha/tamil/confidence_txts/pl_parseq \
    --output_root /ssd_scratch/cvit/lalitha/tamil/confidence_graphs/pl_parseq

# telugu

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/telugu/parseq_ckpts/parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/telugu/test_data/ \
    --test_folder cs/ \
    --output_file /ssd_scratch/cvit/lalitha/telugu/confidence_txts/parseq/cs.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/telugu/parseq_ckpts/parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/telugu/test_data/ \
    --test_folder iiit/ \
    --output_file /ssd_scratch/cvit/lalitha/telugu/confidence_txts/parseq/iiit.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/telugu/parseq_ckpts/parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/telugu/test_data/ \
    --test_folder internet/ \
    --output_file /ssd_scratch/cvit/lalitha/telugu/confidence_txts/parseq/internet.txt \
    --batch_size 64

python 7_conf_dist_graphs_all.py \
    --data_root /ssd_scratch/cvit/lalitha/telugu/confidence_txts/parseq \
    --output_root /ssd_scratch/cvit/lalitha/telugu/confidence_graphs/parseq

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/telugu/parseq_ckpts/pl_parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/telugu/test_data/ \
    --test_folder cs/ \
    --output_file /ssd_scratch/cvit/lalitha/telugu/confidence_txts/pl_parseq/cs.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/telugu/parseq_ckpts/pl_parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/telugu/test_data/ \
    --test_folder iiit/ \
    --output_file /ssd_scratch/cvit/lalitha/telugu/confidence_txts/pl_parseq/iiit.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/telugu/parseq_ckpts/pl_parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/telugu/test_data/ \
    --test_folder internet/ \
    --output_file /ssd_scratch/cvit/lalitha/telugu/confidence_txts/pl_parseq/internet.txt \
    --batch_size 64

python 7_conf_dist_graphs_all.py \
    --data_root /ssd_scratch/cvit/lalitha/telugu/confidence_txts/pl_parseq \
    --output_root /ssd_scratch/cvit/lalitha/telugu/confidence_graphs/pl_parseq

# odia
python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/odia/parseq_ckpts/parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/odia/test_data/ \
    --test_folder iiit/ \
    --output_file /ssd_scratch/cvit/lalitha/odia/confidence_txts/parseq/iiit.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/odia/parseq_ckpts/parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/odia/test_data/ \
    --test_folder internet/ \
    --output_file /ssd_scratch/cvit/lalitha/odia/confidence_txts/parseq/internet.txt \
    --batch_size 64

python 7_conf_dist_graphs_all.py \
    --data_root /ssd_scratch/cvit/lalitha/odia/confidence_txts/parseq \
    --output_root /ssd_scratch/cvit/lalitha/odia/confidence_graphs/parseq

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/odia/parseq_ckpts/pl_parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/odia/test_data/ \
    --test_folder iiit/ \
    --output_file /ssd_scratch/cvit/lalitha/odia/confidence_txts/pl_parseq/iiit.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/odia/parseq_ckpts/pl_parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/odia/test_data/ \
    --test_folder internet/ \
    --output_file /ssd_scratch/cvit/lalitha/odia/confidence_txts/pl_parseq/internet.txt \
    --batch_size 64

python 7_conf_dist_graphs_all.py \
    --data_root /ssd_scratch/cvit/lalitha/odia/confidence_txts/pl_parseq \
    --output_root /ssd_scratch/cvit/lalitha/odia/confidence_graphs/pl_parseq

# punjabi

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/punjabi/parseq_ckpts/parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/punjabi/test_data/ \
    --test_folder cs/ \
    --output_file /ssd_scratch/cvit/lalitha/punjabi/confidence_txts/parseq/cs.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/punjabi/parseq_ckpts/parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/punjabi/test_data/ \
    --test_folder iiit/ \
    --output_file /ssd_scratch/cvit/lalitha/punjabi/confidence_txts/parseq/iiit.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/punjabi/parseq_ckpts/parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/punjabi/test_data/ \
    --test_folder internet/ \
    --output_file /ssd_scratch/cvit/lalitha/punjabi/confidence_txts/parseq/internet.txt \
    --batch_size 64

python 7_conf_dist_graphs_all.py \
    --data_root /ssd_scratch/cvit/lalitha/punjabi/confidence_txts/parseq \
    --output_root /ssd_scratch/cvit/lalitha/punjabi/confidence_graphs/parseq

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/punjabi/parseq_ckpts/pl_parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/punjabi/test_data/ \
    --test_folder cs/ \
    --output_file /ssd_scratch/cvit/lalitha/punjabi/confidence_txts/pl_parseq/cs.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/punjabi/parseq_ckpts/pl_parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/punjabi/test_data/ \
    --test_folder iiit/ \
    --output_file /ssd_scratch/cvit/lalitha/punjabi/confidence_txts/pl_parseq/iiit.txt \
    --batch_size 64

python 5_evaluate_log.py \
    /ssd_scratch/cvit/lalitha/punjabi/parseq_ckpts/pl_parseq.ckpt \
    --data_root /ssd_scratch/cvit/lalitha/punjabi/test_data/ \
    --test_folder internet/ \
    --output_file /ssd_scratch/cvit/lalitha/punjabi/confidence_txts/pl_parseq/internet.txt \
    --batch_size 64

python 7_conf_dist_graphs_all.py \
    --data_root /ssd_scratch/cvit/lalitha/punjabi/confidence_txts/pl_parseq \
    --output_root /ssd_scratch/cvit/lalitha/punjabi/confidence_graphs/pl_parseq