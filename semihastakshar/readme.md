* to train the pl parseq 
1. get the data into paths specified in pl_config.yaml
2. get the word images for unlabeled data and split the data and get the lmdbs
3. modify the labeled data to include img_names
4. modify img_names in lmdb of unlabelled since it is returing error during training 
2. change the commands in train.sh 
3. sbatch train.sh and model gets trained 
4. you'll be left with lmdb data at the end of each cycle along with a bestckpt to continue the training from whichever cycle. 


# CS data for PL Parseq

* get the data into the folders according to the paths in pl_parseq config file 
* get the unlabelled data into the unlabelled data folder such that you have three lmdbs one the internet one and the other two are the train and val lmdbs of crowdsourced data. 
* combine the lmdbs tools/3_merge_lmdb.py
* get the image files tools/lmdb_to_img_fast.py
* delete the lmdb folder and get the lmdb using 6_img_lmdb_imgnames.py
* modify the lmdb folders in data folder to include img_names using tools/4_modify_lmdb.py and rename the folder names accordingly 
* change the train.sh and start the training. 