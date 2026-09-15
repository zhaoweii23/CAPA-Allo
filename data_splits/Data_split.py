#! /usr/bin/env/python3

import os, argparse, pickle, logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'split.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    return logging.getLogger(__name__)

logger = None


def get_protein_esm(feat):
    if feat.get('protein_esm') is not None:
        return feat['protein_esm']
    elif feat.get('protein_esm_segments') is not None:
        segments = feat['protein_esm_segments']
        if segments:
            return np.concatenate(segments, axis=0)
    return None

def collect_proteins(pos_dir, neg_dir):
    samples = []
    for pkl_path in Path(pos_dir).glob("*_features.pkl"):
        try:
            with open(pkl_path, 'rb') as f:
                feat = pickle.load(f)
            if get_protein_esm(feat) is not None:
                samples.append({'pkl_path': str(pkl_path), 'label': 1})
        except Exception as e:
            logger.warning(f"failed {pkl_path}: {e}")
    for pkl_path in Path(neg_dir).glob("*_features.pkl"):
        try:
            with open(pkl_path, 'rb') as f:
                feat = pickle.load(f)
            if get_protein_esm(feat) is not None:
                samples.append({'pkl_path': str(pkl_path), 'label': 0})
        except Exception as e:
            logger.warning(f"failed {pkl_path}: {e}")
    df = pd.DataFrame(samples)
    if df.empty:
        return df
    df['pdb_id'] = df['pkl_path'].apply(lambda x: Path(x).stem.replace('_features', ''))
    return df


def split_by_protein(df, val_split, test_split, seed):
    proteins = df['pdb_id'].unique()
    train_val, test = train_test_split(proteins, test_size=test_split, random_state=seed, stratify=df.groupby('pdb_id')['label'].first())
    train, val = train_test_split(train_val, test_size=val_split/(1-test_split), random_state=seed, stratify=df[df['pdb_id'].isin(train_val)].groupby('pdb_id')['label'].first())
    return train, val, test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos_dir', default='./data/processed/All_Allosteric_feature')
    parser.add_argument('--neg_dir', default='./data/processed/features_full_Orthostic')
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--test_split', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', default='./data_splits')
    args = parser.parse_args()
    global logger
    logger = setup_logging(args.save_dir)
    df = collect_proteins(args.pos_dir, args.neg_dir)
    if df.empty:
        logger.error("no file")
        return
    train_ids, val_ids, test_ids = split_by_protein(df, args.val_split, args.test_split, args.seed)
    for name, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        path = os.path.join(args.save_dir, f'{name}_pdb_ids.txt')
        with open(path, 'w') as f:
            f.write('\n'.join(ids))
        logger.info(f"{name}: {len(ids)} proteins, saved to {path}")
  
    for name, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        sub = df[df['pdb_id'].isin(ids)]
        ratio = sub['label'].mean()
        logger.info(f"{name} allosteric ratio: {ratio:.2%}")

if __name__ == '__main__':
    main()
