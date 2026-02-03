#!/bin/bash

LANGUAGES=("kannada" "malayalam" "oriya" "punjabi" "tamil" "telugu")

for lang in "${LANGUAGES[@]}"; do
  echo "Processing language: $lang"

  python lmdb_to_img.py \
    --lmdbPath /ssd_scratch/cvit/lalitha/data/$lang/uc/test \
    --outputImagePath /ssd_scratch/cvit/lalitha/data/$lang/uc/test_imgs \
    --outputLabelPath /ssd_scratch/cvit/lalitha/data/$lang/uc/gt_file.txt

  python lmdb_to_img.py \
    --lmdbPath /ssd_scratch/cvit/lalitha/data/$lang/words/test \
    --outputImagePath /ssd_scratch/cvit/lalitha/data/$lang/words/test_imgs \
    --outputLabelPath /ssd_scratch/cvit/lalitha/data/$lang/words/gt_file.txt
done
