#!/usr/bin/env python3
import os, sys, argparse, pickle, logging, json, math
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
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    recall_score, precision_score, accuracy_score, matthews_corrcoef
)
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter1d

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('residue_v4_4_chain.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos_dir', type=str, default='./data/processed/features_allosteric_pkl_saprot_0611')
    parser.add_argument('--neg_dir', type=str, default='./data/processed/features_orthostic_pkl_saprot_0611')
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--test_split', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_runs', type=int, default=3)
    parser.add_argument('--run_id', type=int, default=None, help='run a specific run id (0~n_runs-1), if not specified, run all')
    parser.add_argument('--esm_dim', type=int, default=1280)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    parser.add_argument('--proj_dim', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--lambda_rdrop', type=float, default=0.1)
    parser.add_argument('--pos_weight', type=float, default=150.0)
    parser.add_argument('--dynamic_pos_weight', action='store_true', default=False)
    parser.add_argument('--smooth_sigma', type=float, default=2.0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--noise_std', type=float, default=0.01)
    parser.add_argument('--use_swa', action='store_true', default=True)
    parser.add_argument('--swa_start', type=int, default=30)
    parser.add_argument('--swa_lr', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./Model_File/residue_v4_4_chain_exp')
    parser.add_argument('--use_amp', action='store_true')
    return parser.parse_args()

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

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
    elif HAS_TORCH and isinstance(obj, torch.Tensor):
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
            logger.warning(f"Failed to load {pkl_path}: {e}")
    for pkl_path in Path(neg_dir).glob("*_features.pkl"):
        try:
            feat = load_pkl_safe(str(pkl_path))
            esm = get_protein_esm(feat)
            if esm is not None and len(esm) > 0:
                samples.append({'pkl_path': str(pkl_path), 'label': 0})
        except Exception as e:
            logger.warning(f"Failed to load {pkl_path}: {e}")
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
        mask = torch.ones_like(sim) - torch.eye(len(features), device=features.device)
        exp_sim = torch.exp(sim) * mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
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

    def forward(self, x, src_key_padding_mask=None, attn_mask=None):
        x2 = self.norm1(x)
        attn_out, _ = self.self_attn(x2, x2, x2,
                                     key_padding_mask=src_key_padding_mask,
                                     attn_mask=attn_mask)
        x = x + self.drop_path(self.dropout(attn_out))
        x2 = self.norm2(x)
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        x = x + self.drop_path(self.dropout2(ff_out))
        return x

class ChainAwarePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, chain_ids, mask):
        B, L, D = x.shape
        pos = torch.zeros_like(chain_ids)
        for b in range(B):
            valid_mask = mask[b]
            for c in chain_ids[b][valid_mask].unique():
                if c == 0:
                    continue
                chain_pos = (chain_ids[b] == c) & valid_mask
                pos[b][chain_pos] = torch.arange(chain_pos.sum(), device=x.device)
        pos = pos.clamp(0, self.pe.size(0) - 1)
        return self.dropout(x + self.pe[pos])

def build_chain_attn_mask(chain_ids, mask, nhead):
    B, L = chain_ids.shape
    valid = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    same_chain = (chain_ids.unsqueeze(-1) == chain_ids.unsqueeze(-2))
    not_pad = (chain_ids.unsqueeze(-1) != 0) & (chain_ids.unsqueeze(-2) != 0)
    allowed = valid & same_chain & not_pad
    attn_mask = ~allowed
    attn_mask = attn_mask.unsqueeze(1).expand(-1, nhead, -1, -1).reshape(B * nhead, L, L)
    return attn_mask

class ChainEncoder(nn.Module):
    def __init__(self, esm_dim, d_model, nhead, num_layers, dim_feedforward, dropout, drop_path_rate, max_len=20000):
        super().__init__()
        self.proj = nn.Linear(esm_dim, d_model)
        self.pos_enc = ChainAwarePositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = nn.ModuleList([
            TransformerEncoderLayerDP(d_model, nhead, dim_feedforward, dropout, dpr[i])
            for i in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.use_checkpoint = True

    def forward(self, x, mask, chain_ids):
        B, L, _ = x.shape
        nhead = self.layers[0].self_attn.num_heads
        chain_attn_mask = build_chain_attn_mask(chain_ids, mask, nhead)
        x = self.proj(x)
        x = self.pos_enc(x, chain_ids, mask)
        key_padding_mask = ~mask
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, key_padding_mask, chain_attn_mask, use_reentrant=False)
            else:
                x = layer(x, key_padding_mask, chain_attn_mask)
        x = self.final_norm(x)
        return x

class CrossChainAggregator(nn.Module):
    def __init__(self, d_model, num_heads=2, dropout=0.1):
        super().__init__()
        self.chain_self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.chain_norm = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.res_transform = nn.Linear(d_model, d_model)

    def forward(self, h_residue, chain_ids):
        unique_chains = chain_ids.unique()
        num_chains = len(unique_chains)
        if num_chains <= 1:
            return h_residue

        chain_vecs = []
        for c in unique_chains:
            mask_c = (chain_ids == c)
            chain_vecs.append(h_residue[mask_c].mean(dim=0))
        chain_vecs = torch.stack(chain_vecs, dim=0).unsqueeze(0)
        attn_out, _ = self.chain_self_attn(chain_vecs, chain_vecs, chain_vecs)
        chain_vecs = self.chain_norm(chain_vecs + attn_out).squeeze(0)

        enhanced = []
        for idx, c in enumerate(unique_chains):
            mask_c = (chain_ids == c)
            res_feat = h_residue[mask_c]
            global_c = chain_vecs[idx].unsqueeze(0).expand(res_feat.size(0), -1)
            gate_in = torch.cat([res_feat, global_c], dim=-1)
            gamma = self.gate(gate_in)
            out = gamma * res_feat + (1 - gamma) * self.res_transform(global_c)
            enhanced.append(out)

        output = torch.zeros_like(h_residue)
        for idx, c in enumerate(unique_chains):
            mask_c = (chain_ids == c)
            output[mask_c] = enhanced[idx]
        return output

class CrossAttentionPocket(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pocket_query = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, h, mask, return_attention=False):
        B = h.size(0)
        query = self.pocket_query.expand(B, -1, -1)
        attn_out, attn_weights = self.cross_attn(
            query=query, key=h, value=h,
            key_padding_mask=~mask,
            need_weights=True,
            average_attn_weights=True
        )
        if return_attention:
            return self.dropout(attn_out), attn_weights
        return self.dropout(attn_out)

class OptimizedModel(nn.Module):
    def __init__(self, args, geom_feat_dim):
        super().__init__()
        self.chain_encoder = ChainEncoder(
            esm_dim=args.esm_dim, d_model=args.d_model, nhead=args.nhead,
            num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
            dropout=args.dropout, drop_path_rate=args.drop_path_rate
        )
        self.aggregator = CrossChainAggregator(args.d_model, num_heads=2, dropout=args.dropout)
        self.cross_attn = CrossAttentionPocket(args.d_model, nhead=args.nhead, dropout=args.dropout)
        self.layer_norm_cross = nn.LayerNorm(args.d_model)

        self.geom_proj = nn.Linear(geom_feat_dim, args.d_model)
        self.gate_seq = nn.Linear(args.d_model, args.d_model, bias=False)
        self.gate_geom = nn.Linear(args.d_model, args.d_model, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(args.d_model))

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

    def forward(self, seq_esm, mask, geom_feat, chain_ids, return_attention=False):
        B, L = mask.shape
        h = self.chain_encoder(seq_esm, mask, chain_ids)

        h_geom = self.geom_proj(geom_feat)
        gate = torch.sigmoid(self.gate_seq(h) * self.gate_geom(h_geom) + self.gate_bias)
        h = gate * h + (1 - gate) * h_geom

        for b in range(B):
            valid = mask[b]
            c_valid = chain_ids[b][valid]
            if len(c_valid.unique()) > 1:
                h[b][valid] = self.aggregator(h[b][valid], c_valid)

        if return_attention:
            cross_out, attn_weights = self.cross_attn(h, mask, return_attention=True)
        else:
            cross_out = self.cross_attn(h, mask)

        h = h + cross_out.expand(-1, L, -1)
        h = self.layer_norm_cross(h)

        logits = self.classifier(h).squeeze(-1)
        proj = self.projection(h)
        global_feat = (h * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        global_proj = self.global_proj(global_feat)

        if return_attention:
            return logits, proj, global_proj, attn_weights
        return logits, proj, global_proj

class AdaptiveLossWeights:
    def __init__(self, device, num_losses=5):
        self.log_vars = nn.ParameterList([
            nn.Parameter(torch.zeros(1, device=device)) for _ in range(num_losses)
        ])
    def get_weights(self):
        return [torch.exp(-lv) for lv in self.log_vars]
    def parameters(self):
        return list(self.log_vars.parameters())

class ResidueFullGVPDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat = load_pkl_safe(row['pkl_path'])

        esm = get_protein_esm(feat)
        seq_esm = to_tensor(esm)

        full_gvp = feat['full_protein_gvp']
        node_s = to_tensor(full_gvp.node_s)
        node_v = to_tensor(full_gvp.node_v)
        if node_v.dim() == 3:
            node_v_flat = node_v.view(node_v.size(0), -1)
        else:
            node_v_flat = node_v
        geom_feat = torch.cat([node_s, node_v_flat], dim=-1)

        L = seq_esm.size(0)
        labels = torch.zeros(L, dtype=torch.long)
        pocket_indices = feat.get('pocket_indices', [])
        if row['label'] == 1:
            for idx_ in pocket_indices:
                if 0 <= idx_ < L:
                    labels[idx_] = 1

        chain_path = row['pkl_path'].replace('_features.pkl', '_features.chain.npy')
        if os.path.exists(chain_path):
            chain_ids = torch.from_numpy(np.load(chain_path)).long()
        else:
            chain_ids = torch.ones(L, dtype=torch.long)

        return {
            'pdb_id': row['pdb_id'],
            'seq_esm': seq_esm,
            'geom_feat': geom_feat,
            'labels': labels,
            'protein_label': row['label'],
            'chain_ids': chain_ids
        }

def collate_varlen(batch):
    pdb_ids = [item['pdb_id'] for item in batch]
    seq_esm_list = [item['seq_esm'] for item in batch]
    geom_feat_list = [item['geom_feat'] for item in batch]
    labels_list = [item['labels'] for item in batch]
    protein_labels = torch.tensor([item['protein_label'] for item in batch])
    chain_ids_list = [item['chain_ids'] for item in batch]

    max_len = max(seq.size(0) for seq in seq_esm_list)
    B = len(batch)
    padded_seq = torch.zeros(B, max_len, seq_esm_list[0].size(-1))
    padded_geom = torch.zeros(B, max_len, geom_feat_list[0].size(-1))
    padded_labels = torch.zeros(B, max_len, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    padded_chain = torch.zeros(B, max_len, dtype=torch.long)

    for i in range(B):
        L = seq_esm_list[i].size(0)
        padded_seq[i, :L] = seq_esm_list[i]
        padded_geom[i, :L] = geom_feat_list[i]
        padded_labels[i, :L] = labels_list[i]
        mask[i, :L] = True
        padded_chain[i, :L] = chain_ids_list[i]

    return {
        'pdb_ids': pdb_ids,
        'seq_esm': padded_seq,
        'geom_feat': padded_geom,
        'labels': padded_labels,
        'mask': mask,
        'protein_labels': protein_labels,
        'chain_ids': padded_chain
    }

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

def train_epoch(model, ema, optimizer, criterion_dice, criterion_cont, criterion_ssl,
                adaptive_weights, device, grad_clip, args, scaler, loader):
    model.train()
    total_loss = 0.0
    total_bce = 0.0; total_dice = 0.0; total_cont = 0.0; total_ssl = 0.0; total_rdrop = 0.0

    optimizer.zero_grad()
    for i, batch in enumerate(tqdm(loader, desc='Train', leave=False)):
        seq_esm = batch['seq_esm'].to(device)
        geom_feat = batch['geom_feat'].to(device)
        labels = batch['labels'].to(device)
        mask = batch['mask'].to(device)
        chain_ids = batch['chain_ids'].to(device)

        if args.noise_std > 0:
            seq_esm_aug = seq_esm + torch.randn_like(seq_esm) * args.noise_std * mask.unsqueeze(-1)
            geom_feat_aug = geom_feat + torch.randn_like(geom_feat) * args.noise_std * mask.unsqueeze(-1)
        else:
            seq_esm_aug = seq_esm; geom_feat_aug = geom_feat

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            logits1, proj1, global_proj1 = model(seq_esm, mask, geom_feat, chain_ids)
            logits2, proj2, global_proj2 = model(seq_esm_aug, mask, geom_feat_aug, chain_ids)

            valid_mask = mask
            if args.dynamic_pos_weight:
                num_pos = labels[valid_mask].sum()
                num_neg = valid_mask.sum() - num_pos
                pos_weight = (num_neg / (num_pos + 1e-8)).clamp(1, 500)
            else:
                pos_weight = torch.tensor([args.pos_weight]).to(device)

            loss_bce = F.binary_cross_entropy_with_logits(
                logits1[valid_mask], labels[valid_mask].float(),
                pos_weight=pos_weight
            )
            loss_dice = criterion_dice(logits1[valid_mask], labels[valid_mask].float())

            pos_mask = (labels == 1) & valid_mask
            if pos_mask.sum() >= 2:
                pos_feats = proj1[pos_mask]
                loss_cont = criterion_cont(pos_feats, torch.ones(pos_feats.size(0), device=device))
            else:
                loss_cont = torch.tensor(0.0, device=device)

            loss_ssl = criterion_ssl(proj1, global_proj1, mask)

            if args.lambda_rdrop > 0 and valid_mask.sum() > 0:
                p1 = torch.sigmoid(logits1[valid_mask])
                p2 = torch.sigmoid(logits2[valid_mask])
                loss_rdrop = F.kl_div(torch.log(p1 + 1e-8), p2 + 1e-8, reduction='batchmean') + \
                             F.kl_div(torch.log(p2 + 1e-8), p1 + 1e-8, reduction='batchmean')
                loss_rdrop = loss_rdrop * args.lambda_rdrop
            else:
                loss_rdrop = torch.tensor(0.0, device=device)

            weights = adaptive_weights.get_weights()
            loss = (weights[0] * loss_bce + 0.5 * adaptive_weights.log_vars[0] +
                    weights[1] * loss_dice + 0.5 * adaptive_weights.log_vars[1] +
                    weights[2] * loss_cont + 0.5 * adaptive_weights.log_vars[2] +
                    weights[3] * loss_ssl + 0.5 * adaptive_weights.log_vars[3] +
                    weights[4] * loss_rdrop + 0.5 * adaptive_weights.log_vars[4])

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
        total_bce += loss_bce.item()
        total_dice += loss_dice.item()
        total_cont += loss_cont.item()
        total_ssl += loss_ssl.item()
        total_rdrop += loss_rdrop.item()

    n = len(loader)
    logger.info(f"Loss weights: bce={weights[0].item():.3f}, dice={weights[1].item():.3f}, "
                f"cont={weights[2].item():.3f}, ssl={weights[3].item():.3f}, rdrop={weights[4].item():.3f}")
    return (total_loss/n, total_bce/n, total_dice/n, total_cont/n, total_ssl/n, total_rdrop/n)

@torch.no_grad()
def evaluate(model, loader, device, ema=None, smooth_sigma=2.0):
    if ema is not None:
        ema.apply_shadow()
    model.eval()
    all_labels, all_probs_raw, all_probs_smooth = [], [], []
    per_protein_probs_raw, per_protein_labels = [], []
    pdb_ids_list = []

    for batch in tqdm(loader, desc='Eval', leave=False):
        seq_esm = batch['seq_esm'].to(device)
        geom_feat = batch['geom_feat'].to(device)
        labels = batch['labels'].to(device)
        mask = batch['mask'].to(device)
        chain_ids = batch['chain_ids'].to(device)
        pdb_ids = batch['pdb_ids']

        logits, _, _ = model(seq_esm, mask, geom_feat, chain_ids)
        probs = torch.sigmoid(logits)

        for i in range(len(batch['protein_labels'])):
            valid = mask[i].sum()
            if valid == 0:
                continue
            p_raw = probs[i][mask[i]].cpu().numpy()
            l = labels[i][mask[i]].cpu().numpy()

            if smooth_sigma > 0 and len(p_raw) > 3:
                p_smooth = gaussian_filter1d(p_raw, sigma=smooth_sigma)
            else:
                p_smooth = p_raw

            all_probs_raw.append(p_raw)
            all_probs_smooth.append(p_smooth)
            all_labels.append(l)
            per_protein_probs_raw.append(p_smooth)
            per_protein_labels.append(l)
            pdb_ids_list.append(pdb_ids[i])

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

    overall_raw = compute_metrics(np.concatenate(all_labels),
                                  np.concatenate(all_probs_raw))
    overall_raw['count'] = sum(len(a) for a in all_labels)
    overall_smooth = compute_metrics(np.concatenate(all_labels),
                                     np.concatenate(all_probs_smooth))
    overall_smooth['count'] = overall_raw['count']

    per_protein_metrics = []
    for yt, yp, pid in zip(per_protein_labels, per_protein_probs_raw, pdb_ids_list):
        if len(np.unique(yt)) < 2:
            continue
        m = compute_metrics(yt, yp)
        m['pdb_id'] = pid
        per_protein_metrics.append(m)

    if per_protein_metrics:
        pp_avg = {
            'auroc_avg': np.mean([m['auroc'] for m in per_protein_metrics]),
            'auprc_avg': np.mean([m['auprc'] for m in per_protein_metrics]),
            'f1_avg': np.mean([m['f1'] for m in per_protein_metrics]),
            'mcc_avg': np.mean([m['mcc'] for m in per_protein_metrics]),
            'num_proteins': len(per_protein_metrics)
        }
    else:
        pp_avg = {'auroc_avg': 0.0, 'auprc_avg': 0.0, 'f1_avg': 0.0, 'mcc_avg': 0.0, 'num_proteins': 0}

    result = {
        'overall_raw': overall_raw,
        'overall_smooth': overall_smooth,
        'per_protein_avg': pp_avg
    }

    if ema is not None:
        ema.restore()
    return result

def run_experiment(run_id, args, df):
    run_seed = args.seed + run_id
    set_seed(run_seed)
    run_save_dir = os.path.join(args.save_dir, f'run{run_id+1}')
    os.makedirs(run_save_dir, exist_ok=True)

    df_run = split_by_protein(df.copy(), args.val_split, args.test_split, run_seed)

    train_ds = ResidueFullGVPDataset(df_run[df_run['split'] == 'train'])
    val_ds = ResidueFullGVPDataset(df_run[df_run['split'] == 'val'])
    test_ds = ResidueFullGVPDataset(df_run[df_run['split'] == 'test'])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_varlen, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_varlen, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_varlen, num_workers=0, pin_memory=False)

    sample_batch = next(iter(train_loader))
    actual_geom_dim = sample_batch['geom_feat'].size(-1)
    logger.info(f"[Run {run_id+1}] Geometry feature dimension: {actual_geom_dim}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = OptimizedModel(args, geom_feat_dim=actual_geom_dim).to(device)
    ema = EMA(model, decay=0.999) if args.use_swa else None

    criterion_dice = DiceLoss()
    criterion_cont = SupConLoss(temperature=args.temperature)
    criterion_ssl = InfoNCELoss(temperature=args.temperature)

    adaptive_weights = AdaptiveLossWeights(device, num_losses=5)

    optimizer = AdamW([
        {'params': model.parameters()},
        {'params': adaptive_weights.parameters()}
    ], lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = len(train_loader) * args.warmup_epochs // args.grad_accum
    scheduler_cos = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler_warm = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warm, scheduler_cos], milestones=[warmup_steps])

    swa_model, swa_scheduler = None, None
    if args.use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)

    scaler = torch.amp.GradScaler('cuda') if args.use_amp else None
    best_pp_auprc = 0.0
    patience_cnt = 0

    for epoch in range(1, args.epochs + 1):
        logger.info(f"[Run {run_id+1}] Epoch {epoch}/{args.epochs}")
        loss_vals = train_epoch(model, ema, optimizer, criterion_dice, criterion_cont,
                                criterion_ssl, adaptive_weights, device, args.grad_clip, args, scaler, train_loader)
        train_loss, bce, dice, cont, ssl, rdrop = loss_vals
        logger.info(f"Train Loss: {train_loss:.4f} (bce={bce:.4f}, dice={dice:.4f}, cont={cont:.4f}, ssl={ssl:.4f}, rdrop={rdrop:.4f})")

        val_metrics = evaluate(model, val_loader, device, ema=ema, smooth_sigma=args.smooth_sigma)
        pp_avg = val_metrics['per_protein_avg']
        overall_s = val_metrics['overall_smooth']
        logger.info(f"Val Smooth | AUROC: {overall_s['auroc']:.4f} AUPRC: {overall_s['auprc']:.4f} "
                    f"F1: {overall_s['f1']:.4f} Recall: {overall_s['recall']:.4f} Precision: {overall_s['precision']:.4f}")
        logger.info(f"Val Per‑Protein (smoothed) | AUPRC avg: {pp_avg['auprc_avg']:.4f} on {pp_avg['num_proteins']} proteins")

        scheduler.step()
        if swa_model and epoch >= args.swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        if pp_avg['auprc_avg'] > best_pp_auprc:
            best_pp_auprc = pp_avg['auprc_avg']
            patience_cnt = 0
            if ema:
                ema.apply_shadow()
                torch.save(model.state_dict(), os.path.join(run_save_dir, 'best_model.pt'))
                ema.restore()
            else:
                torch.save(model.state_dict(), os.path.join(run_save_dir, 'best_model.pt'))
            logger.info(f"Saved best model (per‑protein AUPRC={best_pp_auprc:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                logger.info("Early stopping triggered.")
                break

    model.load_state_dict(torch.load(os.path.join(run_save_dir, 'best_model.pt'), map_location=device))
    test_metrics = evaluate(model, test_loader, device, ema=None, smooth_sigma=args.smooth_sigma)
    logger.info(f"Test Overall (raw): {json.dumps(test_metrics['overall_raw'], indent=2)}")
    logger.info(f"Test Overall (smooth): {json.dumps(test_metrics['overall_smooth'], indent=2)}")
    logger.info(f"Test Per‑Protein avg: {json.dumps(test_metrics['per_protein_avg'], indent=2)}")
    with open(os.path.join(run_save_dir, 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)

    return {'run': run_id+1, 'test': test_metrics}

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    df = collect_proteins(args.pos_dir, args.neg_dir)
    if df.empty:
        logger.error("No valid samples, exiting.")
        sys.exit(1)

    if args.run_id is not None:
        if args.run_id < 0 or args.run_id >= args.n_runs:
            logger.error(f"run_id {args.run_id} out of range 0~{args.n_runs-1}")
            sys.exit(1)
        logger.info(f"Single run mode: running experiment {args.run_id+1}")
        result = run_experiment(args.run_id, args, df)
        logger.info(f"Run {args.run_id+1} completed, test results saved in {args.save_dir}/run{args.run_id+1}/")
        return

    completed_runs = []
    for r in range(args.n_runs):
        run_dir = os.path.join(args.save_dir, f'run{r+1}')
        if os.path.exists(os.path.join(run_dir, 'test_results.json')):
            completed_runs.append(r)
    unfinished_runs = [r for r in range(args.n_runs) if r not in completed_runs]
    logger.info(f"Completed runs: {completed_runs}, pending: {unfinished_runs}")

    all_results = []
    for r in completed_runs:
        with open(os.path.join(args.save_dir, f'run{r+1}', 'test_results.json'), 'r') as f:
            results = json.load(f)
            all_results.append({'run': r+1, 'test': results})
    for r in unfinished_runs:
        try:
            result = run_experiment(r, args, df)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Run {r+1} failed: {e}")

    all_results = sorted(all_results, key=lambda x: x['run'])
    summary_metrics = ['auroc', 'auprc', 'f1', 'recall', 'precision', 'acc', 'mcc']
    summary = {
        'n_runs': args.n_runs,
        'overall_raw': {m: np.mean([r['test']['overall_raw'][m] for r in all_results]) for m in summary_metrics},
        'overall_smooth': {m: np.mean([r['test']['overall_smooth'][m] for r in all_results]) for m in summary_metrics},
        'per_protein_avg': {m: np.mean([r['test']['per_protein_avg'][m] for r in all_results]) for m in ['auroc_avg','auprc_avg','f1_avg','mcc_avg','num_proteins']}
    }
    with open(os.path.join(args.save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary completed:\n{json.dumps(summary, indent=2)}")

if __name__ == '__main__':
    main()
