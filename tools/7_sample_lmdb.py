import os 
import lmdb
import random

def sample_lmdb(source_path, output_path=None, num_samples=5000, seed=42):
    "Samples a fixed number of entries from LMDB and writes to new LMDB"
    os.makedirs(output_path, exist_ok=True)
    env_in = lmdb.open(source_path, readonly=True, lock=False)
    env_out = lmdb.open(output_path, map_size=1099511627776)

    with env_in.begin() as txn_in:
        total_samples = int(txn_in.get('num-samples'.encode()).decode())
        indices = list(range(1, total_samples + 1))
        random.seed(seed)
        sampled_indices = sorted(random.sample(indices, min(num_samples, total_samples)))

        with env_out.begin(write=True) as txn_out:
            for cnt, i in enumerate(sampled_indices, start=1):
                imageKey = f'image-{i:09d}'.encode()
                labelKey = f'label-{i:09d}'.encode()
                
                imageBin = txn_in.get(imageKey)
                label = txn_in.get(labelKey)

                newImageKey = f'image-{cnt:09d}'.encode()
                newLabelKey = f'label-{cnt:09d}'.encode()

                txn_out.put(newImageKey, imageBin)
                txn_out.put(newLabelKey, label)

                txn_out.put('num-samples'.encode(), str(len(sampled_indices)).encode())

    print(f'Sampled {len(sampled_indices)} entries from {source_path} to {output_path}')
    env_in.close()
    env_out.close()
        

if __name__=='__main__':
    source_path = '/ssd_scratch/cvit/lalitha/data/train/real/full_data'
    output_path = '/ssd_scratch/cvit/lalitha/data/train/real/sampled'
    num_samples = 15000
    seed = 42

    sample_lmdb(source_path, output_path, num_samples, seed)