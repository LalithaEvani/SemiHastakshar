import os
import lmdb
import argparse

def merge_lmdb(lmdb_paths, outputPath):
    """Merges multiple LMDB datasets into a single LMDB dataset."""
    os.makedirs(outputPath, exist_ok=True)
    env_out = lmdb.open(outputPath, map_size=1099511627776)
    cnt = 1
    
    with env_out.begin(write=True) as txn_out:
        for lmdb_path in lmdb_paths:
            env = lmdb.open(lmdb_path, readonly=True, lock=False)
            with env.begin() as txn:
                num_samples = int(txn.get('num-samples'.encode()).decode())
                for i in range(1, num_samples + 1):
                    imageKey = f'image-{i:09d}'.encode()
                    labelKey = f'label-{i:09d}'.encode()
                    
                    imageBin = txn.get(imageKey)
                    label = txn.get(labelKey)
                    
                    newImageKey = f'image-{cnt:09d}'.encode()
                    newLabelKey = f'label-{cnt:09d}'.encode()
                    
                    txn_out.put(newImageKey, imageBin)
                    txn_out.put(newLabelKey, label)
                    cnt += 1
    
        txn_out.put('num-samples'.encode(), str(cnt - 1).encode())
    env_out.close()
    print(f'Merged LMDB dataset saved at {outputPath} with {cnt - 1} samples')

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--lmdb_paths', nargs='+', required=True, help='List of LMDB paths to merge')
    # parser.add_argument('--output_path', required=True, help='Output path for the merged LMDB')
    # args = parser.parse_args()
    lmdb_paths = ['/ssd_scratch/cvit/lalitha/unlabelled/lmdb/cs_train/',
                  '/ssd_scratch/cvit/lalitha/unlabelled/lmdb/cs_val/',
                  '/ssd_scratch/cvit/lalitha/unlabelled/lmdb/internet/']
    output_path = '/ssd_scratch/cvit/lalitha/unlabelled/merged_lmdb'
    
    # merge_lmdb(args.lmdb_paths, args.output_path)
    merge_lmdb(lmdb_paths, output_path)

if __name__ == '__main__':
    main()
