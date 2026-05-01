import os
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

def prepare(seed):
    df = pd.read_csv('fitzpatrick17k_known_code.csv')
    
    # Map fitzpatrick to skin_tone
    # 1, 2, 3 -> light; 4, 5, 6 -> dark
    df['skin_tone'] = df['fitzpatrick'].apply(lambda x: 'light' if int(x) in [1, 2, 3] else 'dark')
    
    # Map nine_partition_label to label 0-8
    labels = sorted(df['nine_partition_label'].unique())
    label_map = {l: i for i, l in enumerate(labels)}
    df['label'] = df['nine_partition_label'].map(label_map)
    
    # Split into 60/20/20 train/val/test
    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=seed, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=temp_df['label'])
    
    # Save
    out_dir = f'./dataset/Fitzpatrick-17k/processed/rand_seed={seed}/split'
    os.makedirs(out_dir, exist_ok=True)
    
    train_df.to_csv(f'{out_dir}/train.csv')
    val_df.to_csv(f'{out_dir}/val.csv')
    test_df.to_csv(f'{out_dir}/test.csv')
    print(f"Saved splits to {out_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    prepare(args.seed)
