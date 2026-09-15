#!/usr/bin/env python3


import os, sys, argparse, pickle, logging, json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    recall_score, precision_score, accuracy_score, matthews_corrcoef
)
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('residue_ablation.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)


# ==================== Argument parsing ====================
def parse_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--pos_dir', type=str, default='./data/processed/features_allosteric_pkl_saprot')
    parser.add_argument('--neg_dir', type=str, default='./data/processed/features_orthostic_pkl_saprot')
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--test_split', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_runs', type=int, default=1, help='Number of repeats per experiment group')
  
    parser.add_argument('--use_geom', type=lambda x: x.lower() in ['true','1','yes'], default=True,
                        help='Whether to use GVP geometric features')
    parser.add_argument('--use_agg', type=lambda x: x.lower() in ['true','1','yes'], default=True,
                        help='Whether to use pocket aggregation features')
    parser.add_argument('--fusion_type', choices=['gate','add','concat'], default='gate',
                        help='Feature fusion strategy (gate / add / concat)')
    parser.add_argument('--use_cross_attn', type=lambda x: x.lower() in ['true','1','yes'], default=True,
                        help='Whether to use cross attention')
    parser.add_argument('--loss_supcon', type=lambda x: x.lower() in ['true','1','yes'], default=True)
    parser.add_argument('--loss_infonce', type=lambda x: x.lower() in ['true','1','yes'], default=True)
    parser.add_argument('--loss_rdrop', type=lambda x: x.lower() in ['true','1','yes'], default=True)
    parser.add_argument('--loss_dice', type=lambda x: x.lower() in ['true','1','yes'], default=True)
    parser.add_argument('--loss_neighbor', type=lambda x: x.lower() in ['true','1','yes'], default=True)
    parser.add_argument('--use_ema', type=lambda x: x.lower() in ['true','1','yes'], default=True,
                        help='Whether to use EMA')
    # Model hyperparameters
    parser.add_argument('--esm_dim', type=int, default=1280)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    parser.add_argument('--proj_dim', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--focal_alpha_pos', type=float, default=0.75)
    parser.add_argument('--focal_alpha_neg', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--neighbor_sigma', type=float, default=5.0)
    # Training
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--noise_std', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--save_dir', type=str, default='./Model_File/residue_ablation_exp')
    parser.add_argument('--use_amp', action='store_true')
    return parser.parse_args()


# ==================== Utility functions ====================
def to_tensor(x, dtype=torch.float):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype=dtype)
    else:
        return torch.tensor(x, dtype=dtype)

def recursive_to_cpu(obj):
    if isinstance(obj, dict):
        return {k: recursive_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_to_cpu(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to_cpu(v) for v in obj)
    elif isinstance(obj, torch.Tensor):
        return obj.cpu()
    else:
        return obj

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_protein_esm(feat):
    if feat.get('protein_esm') is not None:
        return feat['protein_esm']
    elif feat.get('protein_esm_segments') is not None:
        segments = feat['protein_esm_segments']
        if segments:
            return np.concatenate(segments, axis=0)
    return None

def load_pkl_safe(pkl_path):
    with open(pkl_path, 'rb') as f:
        obj = pickle.load(f)
    return recursive_to_cpu(obj)

def collect_proteins(pos_dir, neg_dir):
    samples = []
    for pkl_path in Path(pos_dir).glob("*_features.pkl"):
        try:
            feat = load_pkl_safe(str(pkl_path))
            esm = get_protein_esm(feat)
            if esm is not None and len(esm) > 0:
                samples.append({'pkl_path': str(pkl_path), 'label': 1})
        except Exception as e:
            logger.warning(f"Failed to read {pkl_path}: {e}")
    for pkl_path in Path(neg_dir).glob("*_features.pkl"):
        try:
            feat = load_pkl_safe(str(pkl_path))
            esm = get_protein_esm(feat)
            if esm is not None and len(esm) > 0:
                samples.append({'pkl_path': str(pkl_path), 'label': 0})
        except Exception as e:
            logger.warning(f"Failed to read {pkl_path}: {e}")
    df = pd.DataFrame(samples)
    if df.empty:
        return df
    df['pdb_id'] = df['pkl_path'].apply(lambda x: Path(x).stem.replace('_features', ''))
    logger.info(f"Total valid proteins: {len(df)}, allosteric: {df['label'].sum()}, orthosteric: {len(df)-df['label'].sum()}")
    return df

def split_by_protein(df, val_split, test_split, seed):
    proteins = df['pdb_id'].unique()
    train_val, test = train_test_split(proteins, test_size=test_split, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_split/(1-test_split), random_state=seed)
    df['split'] = 'train'
    df.loc[df['pdb_id'].isin(val), 'split'] = 'val'
    df.loc[df['pdb_id'].isin(test), 'split'] = 'test'
    for s in ['train', 'val', 'test']:
        sub = df[df['split'] == s]
        logger.info(f"{s}: {len(sub)} proteins, allosteric ratio={sub['label'].mean():.2%}")
    return df

def compute_agg_dim_and_keys(pkl_path):
    try:
        feat = load_pkl_safe(pkl_path)
        if 'pocket_aggregated' in feat:
            vals = list(feat['pocket_aggregated'].values())
            if len(vals) > 0:
                return len(vals), None
            return None, None
        if all(k in feat for k in ['spatial_features', 'electron_features', 'aa_features']):
            keys_spatial = sorted(feat['spatial_features'].keys())
            keys_electron = sorted(feat['electron_features'].keys())
            keys_aa = sorted(feat['aa_features'].keys())
            all_keys = keys_spatial + keys_electron + keys_aa
            dim = len(all_keys)
            return dim, (keys_spatial, keys_electron, keys_aa)
    except Exception as e:
        logger.warning(f"Error while computing agg_dim for {pkl_path}: {e}")
    return None, None


# ==================== Loss functions ====================
class FocalLoss(nn.Module):
    def __init__(self, alpha_pos=0.75, alpha_neg=0.25, gamma=2.0):
        super().__init__()
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.gamma = gamma
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha_pos * targets + self.alpha_neg * (1 - targets)
        return (alpha_t * ((1 - pt) ** self.gamma) * bce_loss).mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1. - dice

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    def forward(self, features, labels):
        if len(labels) < 2:
            return torch.tensor(0.0, device=features.device)
        features = F.normalize(features, dim=-1)
        sim = torch.matmul(features, features.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
        mask_same = torch.eq(labels, labels.T).float()
        logits_mask = torch.ones_like(mask_same) - torch.eye(len(labels), device=features.device)
        mask_same = mask_same * logits_mask
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        mean_log_prob_pos = (mask_same * log_prob).sum(dim=1) / mask_same.sum(dim=1).clamp(min=1e-8)
        return -mean_log_prob_pos.mean()

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    def forward(self, residue_feats, global_feats, mask):
        B, L, D = residue_feats.shape
        residue_feats = residue_feats[mask]
        batch_idx = torch.arange(B, device=residue_feats.device).repeat_interleave(mask.sum(dim=1))
        global_sel = global_feats[batch_idx]
        residue_feats = F.normalize(residue_feats, dim=-1)
        global_sel = F.normalize(global_sel, dim=-1)
        pos_sim = (residue_feats * global_sel).sum(dim=-1) / self.temperature
        neg_sim = torch.matmul(residue_feats, global_feats.T) / self.temperature
        neg_sim[torch.arange(len(batch_idx)), batch_idx] = -float('inf')
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(len(batch_idx), dtype=torch.long, device=residue_feats.device)
        return F.cross_entropy(logits, labels)


# ==================== Model components ====================
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor

class TransformerEncoderLayerDP(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, drop_path_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, src_key_padding_mask=None):
        x2 = self.norm1(x)
        attn_out, _ = self.self_attn(x2, x2, x2, key_padding_mask=src_key_padding_mask)
        x = x + self.drop_path(self.dropout(attn_out))
        x2 = self.norm2(x)
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        x = x + self.drop_path(self.dropout2(ff_out))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, esm_dim, d_model, nhead, num_layers, dim_feedforward, dropout, drop_path_rate, max_len=20000):
        super().__init__()
        self.proj = nn.Linear(esm_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = nn.ModuleList([
            TransformerEncoderLayerDP(d_model, nhead, dim_feedforward, dropout, dpr[i])
            for i in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.use_checkpoint = True

    def forward(self, seq_esm, mask):
        x = self.proj(seq_esm)
        x = self.pos_enc(x)
        key_padding_mask = ~mask
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, key_padding_mask, use_reentrant=False)
            else:
                x = layer(x, key_padding_mask)
        return self.final_norm(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])

class CrossAttentionAggregation(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, h_seq, agg_emb, mask):
        query = agg_emb.unsqueeze(1)
        attn_output, _ = self.cross_attn(query=query, key=h_seq, value=h_seq,
                                         key_padding_mask=~mask)
        return self.dropout(attn_output)


# ==================== Ablation model ====================
class AblationModel(nn.Module):
    def __init__(self, args, geom_feat_dim, agg_feat_dim):
        super().__init__()
        self.args = args
        self.use_geom = args.use_geom
        self.use_agg = args.use_agg
        self.fusion_type = args.fusion_type
        self.use_cross_attn = args.use_cross_attn

        # Sequence encoder
        self.encoder = TransformerEncoder(
            esm_dim=args.esm_dim, d_model=args.d_model, nhead=args.nhead,
            num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
            dropout=args.dropout, drop_path_rate=args.drop_path_rate
        )

        # Geometric feature projection (may be disabled)
        if self.use_geom:
            self.geom_proj = nn.Linear(geom_feat_dim, args.d_model)

        # Layers for the chosen fusion strategy
        if self.use_geom:
            if self.fusion_type == 'gate':
                self.gate_seq = nn.Linear(args.d_model, args.d_model, bias=False)
                self.gate_geom = nn.Linear(args.d_model, args.d_model, bias=False)
                self.gate_bias = nn.Parameter(torch.zeros(args.d_model))
            elif self.fusion_type == 'concat':
                self.fuse_proj = nn.Linear(args.d_model * 2, args.d_model)
            # 'add' needs no extra parameters

        # Aggregation feature projection and cross attention
        if self.use_agg:
            self.agg_proj = nn.Linear(agg_feat_dim, args.d_model)
            if self.use_cross_attn:
                self.cross_agg = CrossAttentionAggregation(args.d_model, nhead=args.nhead, dropout=args.dropout)
                self.cross_gate = nn.Linear(args.d_model * 2, args.d_model)
                self.layer_norm = nn.LayerNorm(args.d_model)

        # Classifier and projection heads
        self.classifier = nn.Sequential(
            nn.Linear(args.d_model, args.d_model // 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.d_model // 2, 1)
        )
        self.projection = nn.Sequential(
            nn.Linear(args.d_model, args.proj_dim),
            nn.ReLU(),
            nn.Linear(args.proj_dim, args.proj_dim)
        )
        self.global_proj = nn.Sequential(
            nn.Linear(args.d_model, args.proj_dim),
            nn.ReLU(),
            nn.Linear(args.proj_dim, args.proj_dim)
        )

    def forward(self, seq_esm, mask, geom_feat=None, agg_feat=None):
        h_seq = self.encoder(seq_esm, mask)
        h = h_seq

        if self.use_geom and geom_feat is not None:
            h_geom = self.geom_proj(geom_feat)
            if self.fusion_type == 'gate':
                gate = torch.sigmoid(self.gate_seq(h_seq) * self.gate_geom(h_geom) + self.gate_bias)
                h = gate * h_seq + (1 - gate) * h_geom
            elif self.fusion_type == 'add':
                h = h_seq + h_geom
            elif self.fusion_type == 'concat':
                h = self.fuse_proj(torch.cat([h_seq, h_geom], dim=-1))

        if self.use_agg and agg_feat is not None:
            agg_emb = self.agg_proj(agg_feat)
            if self.use_cross_attn:
                cross_context = self.cross_agg(h, agg_emb, mask)
                cross_context = cross_context.expand(-1, h.size(1), -1)
                gate_cross = torch.sigmoid(self.cross_gate(torch.cat([h, cross_context], dim=-1)))
                h = h + gate_cross * cross_context
                h = self.layer_norm(h)
            else:
                # Simple broadcast addition
                h = h + agg_emb.unsqueeze(1)

        logits = self.classifier(h).squeeze(-1)
        proj = self.projection(h)
        global_feat = (h * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        global_proj = self.global_proj(global_feat)
        return logits, proj, global_proj


# ==================== Dataset (dynamically generates geometric / aggregation features) ====================
class ResidueFullGVPDataset(Dataset):
    def __init__(self, df, neighbor_sigma=5.0, use_geom=True, use_agg=True, agg_dim=None, agg_keys=None):
        self.df = df.reset_index(drop=True)
        self.neighbor_sigma = neighbor_sigma
        self.use_geom = use_geom
        self.use_agg = use_agg
        self.agg_dim = agg_dim
        self.agg_keys = agg_keys

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat = load_pkl_safe(row['pkl_path'])

        esm = get_protein_esm(feat)
        seq_esm = to_tensor(esm)

        # Geometric features
        if self.use_geom:
            full_gvp = feat['full_protein_gvp']
            node_s = to_tensor(full_gvp.node_s)
            node_v = to_tensor(full_gvp.node_v)
            if node_v.dim() == 3:
                node_v_flat = node_v.view(node_v.size(0), -1)
            else:
                node_v_flat = node_v
            geom_feat = torch.cat([node_s, node_v_flat], dim=-1)
        else:
            geom_feat = torch.zeros(seq_esm.size(0), 0)  # Placeholder, not used in computation

        # Aggregation features
        if self.use_agg:
            if 'pocket_aggregated' in feat:
                agg_dict = feat['pocket_aggregated']
                agg_feat = torch.tensor(list(agg_dict.values()), dtype=torch.float)
            elif self.agg_keys is not None:
                keys_spatial, keys_electron, keys_aa = self.agg_keys
                values = []
                for k in keys_spatial:
                    values.append(feat['spatial_features'].get(k, 0.0))
                for k in keys_electron:
                    values.append(feat['electron_features'].get(k, 0.0))
                for k in keys_aa:
                    values.append(feat['aa_features'].get(k, 0.0))
                agg_feat = torch.tensor(values, dtype=torch.float)
            elif self.agg_dim is not None:
                agg_feat = torch.zeros(self.agg_dim, dtype=torch.float)
            else:
                agg_feat = torch.zeros(0, dtype=torch.float)
        else:
            agg_feat = torch.zeros(0, dtype=torch.float)

        L = seq_esm.size(0)
        labels = torch.full((L,), 2, dtype=torch.long)
        pocket_indices = feat.get('pocket_indices', [])
        for idx_ in pocket_indices:
            if 0 <= idx_ < L:
                labels[idx_] = row['label']

        neighbor_indices = feat.get('pocket_neighbor_indices', [])
        neighbor_mask = torch.zeros(L, dtype=torch.bool)
        neighbor_target = torch.zeros(L, dtype=torch.float)
        if len(pocket_indices) > 0 and len(neighbor_indices) > 0:
            pocket_arr = torch.tensor(pocket_indices, dtype=torch.long)
            for n_idx in neighbor_indices:
                if 0 <= n_idx < L:
                    neighbor_mask[n_idx] = True
                    dist = (pocket_arr - n_idx).abs().min().float()
                    target = 0.5 * torch.exp(-dist / self.neighbor_sigma)
                    neighbor_target[n_idx] = target

        return {
            'pdb_id': row['pdb_id'],
            'seq_esm': seq_esm,
            'geom_feat': geom_feat,
            'agg_feat': agg_feat,
            'labels': labels,
            'neighbor_mask': neighbor_mask,
            'neighbor_target': neighbor_target,
            'protein_label': row['label']
        }


def collate_varlen(batch):
    pdb_ids = [item['pdb_id'] for item in batch]
    seq_esm_list = [item['seq_esm'] for item in batch]
    geom_feat_list = [item['geom_feat'] for item in batch]
    agg_feat_list = [item['agg_feat'] for item in batch]
    labels_list = [item['labels'] for item in batch]
    neighbor_mask_list = [item['neighbor_mask'] for item in batch]
    neighbor_target_list = [item['neighbor_target'] for item in batch]
    protein_labels = torch.tensor([item['protein_label'] for item in batch])

    max_len = max(seq.size(0) for seq in seq_esm_list)
    B = len(batch)
    padded_seq = torch.zeros(B, max_len, seq_esm_list[0].size(-1))
    # Geometric feature dim may be 0; get first non-zero dim
    geom_dim = max(g.size(-1) for g in geom_feat_list) if geom_feat_list else 0
    padded_geom = torch.zeros(B, max_len, geom_dim)
    agg_dim = max(a.size(-1) for a in agg_feat_list) if agg_feat_list else 0
    padded_agg = torch.zeros(B, agg_dim) if agg_dim > 0 else torch.zeros(B, 0)
    padded_labels = torch.full((B, max_len), 2, dtype=torch.long)
    padded_neighbor_mask = torch.zeros(B, max_len, dtype=torch.bool)
    padded_neighbor_target = torch.zeros(B, max_len, dtype=torch.float)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i in range(B):
        L = seq_esm_list[i].size(0)
        padded_seq[i, :L] = seq_esm_list[i]
        if geom_dim > 0 and geom_feat_list[i].size(-1) == geom_dim:
            padded_geom[i, :L] = geom_feat_list[i]
        if agg_dim > 0 and agg_feat_list[i].size(-1) == agg_dim:
            padded_agg[i] = agg_feat_list[i]
        padded_labels[i, :L] = labels_list[i]
        padded_neighbor_mask[i, :L] = neighbor_mask_list[i]
        padded_neighbor_target[i, :L] = neighbor_target_list[i]
        mask[i, :L] = True

    return {
        'pdb_ids': pdb_ids,
        'seq_esm': padded_seq,
        'geom_feat': padded_geom,
        'agg_feat': padded_agg,
        'labels': padded_labels,
        'neighbor_mask': padded_neighbor_mask,
        'neighbor_target': padded_neighbor_target,
        'mask': mask,
        'protein_labels': protein_labels
    }


# ==================== EMA ====================
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}
        self.backup = {}

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ==================== Adaptive loss weights (dynamically adjust active losses) ====================
class AdaptiveLossWeights:
    def __init__(self, device, loss_mask):
        # loss_mask: list of 6 bools corresponding to [focal, dice, supcon, infonce, neighbor, rdrop]
        self.mask = loss_mask
        self.log_vars = nn.ParameterList([
            nn.Parameter(torch.zeros(1, device=device)) if use else None
            for use in loss_mask
        ])

    def get_weights(self):
        return [torch.exp(-lv) if lv is not None else torch.tensor(1.0) for lv in self.log_vars]

    def parameters(self):
        return [lv for lv in self.log_vars if lv is not None]


# ==================== Training and evaluation ====================
def train_epoch(model, ema, optimizer, criterion_bce, criterion_dice, criterion_cont, criterion_ssl,
                adaptive_weights, loss_mask, device, grad_clip, args, scaler, loader):
    model.train()
    total_loss = 0.0
    loss_track = [0.0] * 6  # focal, dice, supcon, infonce, neighbor, rdrop

    optimizer.zero_grad()
    for i, batch in enumerate(tqdm(loader, desc='Train', leave=False)):
        seq_esm = batch['seq_esm'].to(device)
        geom_feat = batch['geom_feat'].to(device) if args.use_geom else None
        agg_feat = batch['agg_feat'].to(device) if args.use_agg else None
        labels = batch['labels'].to(device)
        neighbor_mask = batch['neighbor_mask'].to(device)
        neighbor_target = batch['neighbor_target'].to(device)
        mask = batch['mask'].to(device)

        # Data augmentation (R-Drop)
        if args.noise_std > 0 and loss_mask[5]:  # only when R-Drop is enabled
            seq_esm_aug = seq_esm + torch.randn_like(seq_esm) * args.noise_std * mask.unsqueeze(-1)
            geom_feat_aug = geom_feat + torch.randn_like(geom_feat) * args.noise_std * mask.unsqueeze(-1) if geom_feat is not None else None
        else:
            seq_esm_aug, geom_feat_aug = seq_esm, geom_feat

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            logits1, proj1, global_proj1 = model(seq_esm, mask, geom_feat, agg_feat)
            if loss_mask[5]:
                logits2, proj2, global_proj2 = model(seq_esm_aug, mask, geom_feat_aug, agg_feat)

            valid_mask = (labels != 2) & mask

            # 1. Focal Loss (always computed)
            loss_bce = criterion_bce(logits1[valid_mask], labels[valid_mask].float()) if valid_mask.sum() > 0 else torch.tensor(0.0, device=device)
            loss_track[0] += loss_bce.item()

            # 2. Dice Loss
            loss_dice = criterion_dice(logits1[valid_mask], labels[valid_mask].float()) if (loss_mask[1] and valid_mask.sum() > 0) else torch.tensor(0.0, device=device)
            loss_track[1] += loss_dice.item()

            # 3. SupCon
            if loss_mask[2] and valid_mask.sum() >= 2:
                loss_cont = criterion_cont(proj1[valid_mask], labels[valid_mask])
            else:
                loss_cont = torch.tensor(0.0, device=device)
            loss_track[2] += loss_cont.item()

            # 4. InfoNCE
            if loss_mask[3]:
                loss_ssl = criterion_ssl(proj1, global_proj1, mask)
            else:
                loss_ssl = torch.tensor(0.0, device=device)
            loss_track[3] += loss_ssl.item()

            # 5. Neighbor
            neighbor_valid = neighbor_mask & mask
            if loss_mask[4] and neighbor_valid.sum() > 0:
                loss_neighbor = F.mse_loss(torch.sigmoid(logits1[neighbor_valid]), neighbor_target[neighbor_valid])
            else:
                loss_neighbor = torch.tensor(0.0, device=device)
            loss_track[4] += loss_neighbor.item()

            # 6. R-Drop
            if loss_mask[5] and valid_mask.sum() > 0:
                p1 = torch.sigmoid(logits1[valid_mask])
                p2 = torch.sigmoid(logits2[valid_mask])
                loss_rdrop = (F.kl_div(torch.log(p1 + 1e-8), p2 + 1e-8, reduction='batchmean') +
                             F.kl_div(torch.log(p2 + 1e-8), p1 + 1e-8, reduction='batchmean')) * 0.5
            else:
                loss_rdrop = torch.tensor(0.0, device=device)
            loss_track[5] += loss_rdrop.item()

            # Combine losses; adaptive weights only apply to activated losses
            weights = adaptive_weights.get_weights()
            loss_terms = [loss_bce, loss_dice, loss_cont, loss_ssl, loss_neighbor, loss_rdrop]
            loss = 0.0
            for idx, (term, w, use) in enumerate(zip(loss_terms, weights, loss_mask)):
                if use:
                    loss = loss + w * term + 0.5 * adaptive_weights.log_vars[idx] if adaptive_weights.log_vars[idx] is not None else w * term

        loss = loss / args.grad_accum
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % args.grad_accum == 0 or (i + 1 == len(loader)):
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update()

        total_loss += loss.item() * args.grad_accum

    n = len(loader)
    avg_loss = [total_loss / n] + [x / n for x in loss_track]
    logger.info(f"Loss weights: " + ", ".join([f"{w.item():.3f}" if w is not None else "fixed" for w in weights]))
    return avg_loss  # total, focal, dice, supcon, infonce, neighbor, rdrop


@torch.no_grad()
def evaluate(model, loader, device, args, ema=None, smooth_window=3):
    if ema is not None:
        ema.apply_shadow()
    model.eval()
    all_labels, all_probs = [], []
    allo_labels, allo_probs = [], []
    ortho_labels, ortho_probs = [], []
    per_protein_probs, per_protein_labels = [], []
    pdb_ids_list = []

    for batch in tqdm(loader, desc='Eval', leave=False):
        seq_esm = batch['seq_esm'].to(device)
        geom_feat = batch['geom_feat'].to(device) if args.use_geom else None
        agg_feat = batch['agg_feat'].to(device) if args.use_agg else None
        labels = batch['labels'].to(device)
        mask = batch['mask'].to(device)
        protein_labels = batch['protein_labels'].to(device)
        pdb_ids = batch['pdb_ids']

        logits, _, _ = model(seq_esm, mask, geom_feat, agg_feat)
        probs = torch.sigmoid(logits)
        valid_mask = (labels != 2) & mask

        for i in range(len(protein_labels)):
            if valid_mask[i].sum() == 0:
                continue
            p = probs[i][valid_mask[i]].cpu().numpy()
            l = labels[i][valid_mask[i]].cpu().numpy()
            if smooth_window > 1 and len(p) > smooth_window:
                p = np.convolve(p, np.ones(smooth_window)/smooth_window, mode='same')
            all_probs.append(p)
            all_labels.append(l)
            per_protein_probs.append(p)
            per_protein_labels.append(l)
            pdb_ids_list.append(pdb_ids[i])
            if protein_labels[i] == 1:
                allo_probs.append(p)
                allo_labels.append(l)
            else:
                ortho_probs.append(p)
                ortho_labels.append(l)

    def compute_metrics(y_true, y_prob):
        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            return {'auroc': 0.5, 'auprc': 0.0, 'f1': 0.0, 'recall': 0.0,
                    'precision': 0.0, 'acc': 0.0, 'mcc': 0.0}
        pred_binary = (y_prob >= 0.5).astype(int)
        return {
            'auroc': roc_auc_score(y_true, y_prob),
            'auprc': average_precision_score(y_true, y_prob),
            'f1': f1_score(y_true, pred_binary, zero_division=0),
            'recall': recall_score(y_true, pred_binary, zero_division=0),
            'precision': precision_score(y_true, pred_binary, zero_division=0),
            'acc': accuracy_score(y_true, pred_binary),
            'mcc': matthews_corrcoef(y_true, pred_binary)
        }

    overall = compute_metrics(np.concatenate(all_labels) if all_labels else [],
                              np.concatenate(all_probs) if all_probs else [])
    overall['count'] = sum(len(a) for a in all_labels)
    allo_metrics = compute_metrics(np.concatenate(allo_labels) if allo_labels else [],
                                   np.concatenate(allo_probs) if allo_probs else []) if allo_labels else None
    ortho_metrics = compute_metrics(np.concatenate(ortho_labels) if ortho_labels else [],
                                    np.concatenate(ortho_probs) if ortho_probs else []) if ortho_labels else None

    per_protein_metrics = []
    for yt, yp, pid in zip(per_protein_labels, per_protein_probs, pdb_ids_list):
        if len(yt) == 0:
            continue
        m = compute_metrics(yt, yp)
        m['pdb_id'] = pid
        per_protein_metrics.append(m)

    pp_avg = {
        'auroc_avg': np.mean([m['auroc'] for m in per_protein_metrics]) if per_protein_metrics else 0.0,
        'auprc_avg': np.mean([m['auprc'] for m in per_protein_metrics]) if per_protein_metrics else 0.0,
        'f1_avg': np.mean([m['f1'] for m in per_protein_metrics]) if per_protein_metrics else 0.0,
        'mcc_avg': np.mean([m['mcc'] for m in per_protein_metrics]) if per_protein_metrics else 0.0,
        'num_proteins': len(per_protein_metrics)
    }

    result = {
        'overall': overall,
        'allosteric': allo_metrics,
        'orthosteric': ortho_metrics,
        'per_protein_avg': pp_avg
    }
    if ema is not None:
        ema.restore()
    return result


# ==================== Single experiment run (with ablation config saving) ====================
def run_experiment(run_id, args, df, loss_mask):
    run_seed = args.seed + run_id
    set_seed(run_seed)
    run_save_dir = os.path.join(args.save_dir, f'run{run_id+1}')
    os.makedirs(run_save_dir, exist_ok=True)

    # Save ablation config
    ablation_config = {
        'use_geom': args.use_geom,
        'use_agg': args.use_agg,
        'fusion_type': args.fusion_type,
        'use_cross_attn': args.use_cross_attn,
        'loss_supcon': args.loss_supcon,
        'loss_infonce': args.loss_infonce,
        'loss_rdrop': args.loss_rdrop,
        'loss_dice': args.loss_dice,
        'loss_neighbor': args.loss_neighbor,
        'use_ema': args.use_ema
    }
    with open(os.path.join(run_save_dir, 'ablation_config.json'), 'w') as f:
        json.dump(ablation_config, f, indent=2)

    df_run = split_by_protein(df.copy(), args.val_split, args.test_split, run_seed)
    train_df = df_run[df_run['split'] == 'train']

    # Infer aggregation feature dimension (if needed)
    agg_dim = None
    agg_keys = None
    if args.use_agg:
        for _, row in train_df.iterrows():
            dim, keys = compute_agg_dim_and_keys(row['pkl_path'])
            if dim is not None:
                agg_dim = dim
                agg_keys = keys
                break
        if agg_dim is None:
            logger.warning("Could not infer aggregation feature dim, using 0")
            agg_dim = 0
    else:
        agg_dim = 0

    logger.info(f"Pocket aggregation feature dim: {agg_dim}")

    train_ds = ResidueFullGVPDataset(train_df, neighbor_sigma=args.neighbor_sigma,
                                     use_geom=args.use_geom, use_agg=args.use_agg,
                                     agg_dim=agg_dim, agg_keys=agg_keys)
    val_ds = ResidueFullGVPDataset(df_run[df_run['split']=='val'], neighbor_sigma=args.neighbor_sigma,
                                   use_geom=args.use_geom, use_agg=args.use_agg,
                                   agg_dim=agg_dim, agg_keys=agg_keys)
    test_ds = ResidueFullGVPDataset(df_run[df_run['split']=='test'], neighbor_sigma=args.neighbor_sigma,
                                    use_geom=args.use_geom, use_agg=args.use_agg,
                                    agg_dim=agg_dim, agg_keys=agg_keys)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_varlen, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_varlen, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_varlen, num_workers=0, pin_memory=False)

    sample_batch = next(iter(train_loader))
    geom_feat_dim = sample_batch['geom_feat'].size(-1) if args.use_geom else 0
    logger.info(f"Geometric feature dim: {geom_feat_dim}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = AblationModel(args, geom_feat_dim=geom_feat_dim, agg_feat_dim=agg_dim).to(device)
    ema = EMA(model, decay=0.999) if args.use_ema else None

    criterion_bce = FocalLoss(alpha_pos=args.focal_alpha_pos, alpha_neg=args.focal_alpha_neg, gamma=args.focal_gamma)
    criterion_dice = DiceLoss() if loss_mask[1] else None
    criterion_cont = SupConLoss(temperature=args.temperature) if loss_mask[2] else None
    criterion_ssl = InfoNCELoss(temperature=args.temperature) if loss_mask[3] else None

    adaptive_weights = AdaptiveLossWeights(device, loss_mask)
    optimizer = AdamW([
        {'params': model.parameters()},
        {'params': adaptive_weights.parameters()}
    ], lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = len(train_loader) * args.warmup_epochs // args.grad_accum
    scheduler_cos = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler_warm = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warm, scheduler_cos], milestones=[warmup_steps])

    scaler = torch.amp.GradScaler('cuda') if args.use_amp else None
    best_auprc = 0.0
    patience_cnt = 0

    for epoch in range(1, args.epochs + 1):
        logger.info(f"[Run {run_id+1}] Epoch {epoch}/{args.epochs}")
        avg_losses = train_epoch(model, ema, optimizer, criterion_bce, criterion_dice,
                                 criterion_cont, criterion_ssl, adaptive_weights, loss_mask,
                                 device, args.grad_clip, args, scaler, train_loader)
        train_loss = avg_losses[0]
        logger.info(f"Train Loss: {train_loss:.4f} (Focal:{avg_losses[1]:.4f}, Dice:{avg_losses[2]:.4f}, "
                    f"SupCon:{avg_losses[3]:.4f}, InfoNCE:{avg_losses[4]:.4f}, Neigh:{avg_losses[5]:.4f}, "
                    f"R-Drop:{avg_losses[6]:.4f})")

        val_metrics = evaluate(model, val_loader, device, args, ema=ema, smooth_window=3)
        val_overall = val_metrics['overall']
        logger.info(f"Val AUROC: {val_overall['auroc']:.4f} AUPRC: {val_overall['auprc']:.4f} "
                    f"F1: {val_overall['f1']:.4f} MCC: {val_overall['mcc']:.4f} "
                    f"Per-protein AUPRC avg: {val_metrics['per_protein_avg']['auprc_avg']:.4f}")

        scheduler.step()

        if val_overall['auprc'] > best_auprc:
            best_auprc = val_overall['auprc']
            patience_cnt = 0
            if ema is not None:
                ema.apply_shadow()
                torch.save(model.state_dict(), os.path.join(run_save_dir, 'best_model.pt'))
                ema.restore()
            else:
                torch.save(model.state_dict(), os.path.join(run_save_dir, 'best_model.pt'))
            logger.info(f"Saved best model (AUPRC={best_auprc:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                logger.info("Early stopping")
                break

    # Load best model and test
    model.load_state_dict(torch.load(os.path.join(run_save_dir, 'best_model.pt'), map_location=device))
    logger.info(f"[Run {run_id+1}] Loaded best model.")
    test_metrics = evaluate(model, test_loader, device, args, ema=None, smooth_window=3)
    logger.info(f"Test Overall: {json.dumps(test_metrics['overall'], indent=2)}")
    logger.info(f"Test Per-protein avg: {json.dumps(test_metrics['per_protein_avg'], indent=2)}")
    with open(os.path.join(run_save_dir, 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)

    return test_metrics


# ==================== Main function ====================
def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # Build loss mask [focal, dice, supcon, infonce, neighbor, rdrop]
    loss_mask = [True,
                 args.loss_dice,
                 args.loss_supcon,
                 args.loss_infonce,
                 args.loss_neighbor,
                 args.loss_rdrop]

    logger.info("Ablation config:")
    logger.info(f"  use_geom: {args.use_geom}")
    logger.info(f"  use_agg: {args.use_agg}")
    logger.info(f"  fusion_type: {args.fusion_type}")
    logger.info(f"  use_cross_attn: {args.use_cross_attn}")
    logger.info(f"  loss_mask: {loss_mask}")
    logger.info(f"  use_ema: {args.use_ema}")

    df = collect_proteins(args.pos_dir, args.neg_dir)
    if df.empty:
        logger.error("No valid samples found, exiting.")
        sys.exit(1)

    all_results = []
    for run_id in range(args.n_runs):
        logger.info(f"\n{'='*60}\nStarting experiment Run {run_id+1}/{args.n_runs}\n{'='*60}")
        metrics = run_experiment(run_id, args, df, loss_mask)
        all_results.append(metrics)

    summary = {
        'config': {
            'use_geom': args.use_geom,
            'use_agg': args.use_agg,
            'fusion_type': args.fusion_type,
            'use_cross_attn': args.use_cross_attn,
            'loss_mask': loss_mask,
            'use_ema': args.use_ema,
            'n_runs': args.n_runs,
            'base_seed': args.seed
        },
        'results': all_results,
        'overall_avg': {
            metric: np.mean([r['overall'][metric] for r in all_results])
            for metric in ['auroc', 'auprc', 'f1', 'recall', 'precision', 'acc', 'mcc']
        }
    }

    with open(os.path.join(args.save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Experiment finished, results saved to {args.save_dir}/summary.json")


if __name__ == '__main__':
    main()
