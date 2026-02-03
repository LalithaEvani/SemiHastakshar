#!/bin/bash
#SBATCH -A evanilalitha
#SBATCH --nodelist=gnode049
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=1-00:00:00
#SBATCH --output=../bash_outputs/modify_lmdb_pl_parseq.txt


conda activate parseq
cd /home2/evanilalitha/10_generalized_hindi_HTR/generalized_hindi_HTR/tools/

# Hindi
python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/hindi/test_data/cs \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/hindi/test_data/cs_new
    
python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/hindi/test_data/iiit \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/hindi/test_data/iiit_new

python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/hindi/test_data/internet \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/hindi/test_data/internet_new

# Bengali

python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/bengali/test_data/cs \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/bengali/test_data/cs_new
    
python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/bengali/test_data/iiit \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/bengali/test_data/iiit_new

python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/bengali/test_data/internet \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/bengali/test_data/internet_new
      
# TElugu
python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/telugu/test_data/cs \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/telugu/test_data/cs_new
    
python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/telugu/test_data/iiit \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/telugu/test_data/iiit_new

python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/telugu/test_data/internet \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/telugu/test_data/internet_new
# kannada
python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/kannada/test_data/cs \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/kannada/test_data/cs_new
    
python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/kannada/test_data/iiit \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/kannada/test_data/iiit_new

python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/kannada/test_data/internet \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/kannada/test_data/internet_new
# malayalam
python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/malayalam/test_data/cs \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/malayalam/test_data/cs_new
    
python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/malayalam/test_data/iiit \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/malayalam/test_data/iiit_new

python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/malayalam/test_data/internet \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/malayalam/test_data/internet_new
# tamil
python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/tamil/test_data/cs \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/tamil/test_data/cs_new
    
python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/tamil/test_data/iiit \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/tamil/test_data/iiit_new

python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/tamil/test_data/internet \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/tamil/test_data/internet_new
# odia
python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/odia/test_data/cs \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/odia/test_data/cs_new
    
python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/odia/test_data/iiit \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/odia/test_data/iiit_new

python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/odia/test_data/internet \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/odia/test_data/internet_new
# punjabi
python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/punjabi/test_data/cs \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/punjabi/test_data/cs_new
    
python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/punjabi/test_data/iiit \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/punjabi/test_data/iiit_new

python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/punjabi/test_data/internet \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/punjabi/test_data/internet_new
# gujarati
python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/gujarati/test_data/cs \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/gujarati/test_data/cs_new
    
python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/gujarati/test_data/iiit \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/gujarati/test_data/iiit_new

python 4_modify_lmdb.py --inputLmdbPath /ssd_scratch/cvit/lalitha/gujarati/test_data/internet \
      --outputLmdbPath /ssd_scratch/cvit/lalitha/gujarati/test_data/internet_new
