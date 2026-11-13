#!/usr/bin/env python3


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from Bio import AlignIO
from Bio.PDB import PDBParser, PDBIO, NeighborSearch
from sklearn.metrics import precision_recall_curve, average_precision_score


sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12


MSA_FILE = "aligned_sequences.aln"

APO_PDB = "data/raw/Gluk2/9C60.pdb"
APO_PRED_JSON = "Gluk2/predictions_per_protein/9C60_prediction.json"
APO_CHAIN_IDS = ['A', 'B', 'C', 'D']

COMPLEX_PDB = "data/raw/Gluk2/8FWQ.pdb"
COMPLEX_CHAIN_RESIDUES = {
    'B': [519, 531, 532, 533, 534, 535, 536, 761, 762, 763, 782, 783, 786, 791],
    'C': [519, 531, 532, 533, 534, 535, 536, 761, 762, 763, 782, 783, 786, 791]
}

# COMPLEX_PDB = "data/raw/Gluk2/5L1F.pdb"
# COMPLEX_CHAIN_RESIDUES = {
#     'A': [611],
#     'B': [510, 511, 512, 516, 517, 518, 519, 520, 526, 616, 619, 620, 623, 788, 790, 791]
# }

PROB_THRESHOLD = 0.5
TOP_K = 50
N_PERM = 10000          
RANDOM_SEED = 42

OUTPUT_DIR = Path("evaluation_results_8FWQ")
OUTPUT_DIR.mkdir(exist_ok=True)


def residue_numbers_to_sequence_indices(pdb_file, chain_id, residue_numbers):

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb', pdb_file)
    residues = []
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for res in chain:
                    if res.get_id()[0] == ' ':
                        residues.append(res)
    residues.sort(key=lambda r: (r.get_id()[1], r.get_id()[2]))
    resseq_to_idx = {r.get_id()[1]: i for i, r in enumerate(residues)}
    return [resseq_to_idx[num] for num in residue_numbers if num in resseq_to_idx]


def map_positions(alignment, query_id, target_id, query_positions):

    query_record = None
    target_record = None
    for record in alignment:
        if record.id == query_id:
            query_record = record
        elif record.id == target_id:
            target_record = record
    if query_record is None or target_record is None:
        raise ValueError(f"sequence ID could not found: {query_id} or {target_id}")

    q_pos = -1
    t_pos = -1
    q_to_t = {}
    for q_char, t_char in zip(str(query_record.seq), str(target_record.seq)):
        if q_char != '-':
            q_pos += 1
        if t_char != '-':
            t_pos += 1
        if q_char != '-' and t_char != '-':
            q_to_t[q_pos] = t_pos

    return [q_to_t[pos] for pos in query_positions if pos in q_to_t]


def get_chain_lengths(pdb_file, chain_ids):

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb', pdb_file)
    lengths = {}
    for model in structure:
        for chain in model:
            if chain.id in chain_ids:
                count = sum(1 for res in chain if res.get_id()[0] == ' ')
                lengths[chain.id] = count
    return lengths


def evaluate(probs, known_indices, threshold=0.5, top_k=50):
    if len(probs) == 0 or len(known_indices) == 0:
        return None

    pred_set = set(np.where(probs >= threshold)[0])
    known_set = set(known_indices)
    inter = pred_set & known_set
    union = pred_set | known_set

    precision = len(inter) / len(pred_set) if pred_set else 0.0
    recall = len(inter) / len(known_set) if known_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = len(inter) / len(union) if union else 0.0

    L = len(probs)
    order = np.argsort(-probs)
    rank = np.empty_like(order)
    rank[order] = np.arange(1, L + 1)
    top_k_set = set(order[:top_k])
    top_recall = len(top_k_set & known_set) / len(known_set)
    mean_percentile = np.mean(rank[known_indices] / L)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'top_k_recall': top_recall,
        'mean_rank_percentile': mean_percentile,
        'known_count': len(known_set),
        'pred_count': len(pred_set)
    }


def compute_ranks(probs):

    L = len(probs)
    order = np.argsort(-probs)
    rank = np.empty(L, dtype=int)
    rank[order] = np.arange(1, L + 1)
    return rank, order




def permutation_test_rank_percentile(probs, known_indices,
                                     n_perm=N_PERM, seed=RANDOM_SEED):

    rng = np.random.default_rng(seed)
    L = len(probs)
    n_known = len(known_indices)
    rank, _ = compute_ranks(probs)

    observed = np.mean(rank[known_indices] / L)

    null_dist = np.empty(n_perm)
    for i in range(n_perm):
        rand_idx = rng.choice(L, size=n_known, replace=False)
        null_dist[i] = np.mean(rank[rand_idx] / L)

    # p_value = np.mean(null_dist <= observed)
    p_value = (np.sum(null_dist <= observed) + 1) / (n_perm + 1)
    z_score = (observed - null_dist.mean()) / (null_dist.std() + 1e-12)
    return observed, p_value, z_score, null_dist


def permutation_test_topk_recall(probs, known_indices, top_k=TOP_K,
                                 n_perm=N_PERM, seed=RANDOM_SEED):

    rng = np.random.default_rng(seed)
    L = len(probs)
    n_known = len(known_indices)
    _, order = compute_ranks(probs)
    top_k_set = set(order[:top_k])
    known_set = set(known_indices)

    observed = len(top_k_set & known_set) / n_known

    null_dist = np.empty(n_perm)
    for i in range(n_perm):
        rand_idx = rng.choice(L, size=n_known, replace=False)
        null_dist[i] = len(set(rand_idx) & top_k_set) / n_known

    # p_value = np.mean(null_dist >= observed)
    p_value = (np.sum(null_dist >= observed) + 1) / (n_perm + 1)
    z_score = (observed - null_dist.mean()) / (null_dist.std() + 1e-12)
    return observed, p_value, z_score, null_dist


def write_known_site_pdb(apo_pdb, global_indices, output_pdb):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('apo', apo_pdb)
    all_residues = []
    for model in structure:
        for chain in model:
            if chain.id in APO_CHAIN_IDS:
                for res in chain:
                    if res.get_id()[0] == ' ':
                        all_residues.append(res)
    all_residues.sort(
        key=lambda r: (APO_CHAIN_IDS.index(r.get_parent().id),
                       r.get_id()[1], r.get_id()[2]))

    known_set = set(global_indices)
    for i, res in enumerate(all_residues):
        bval = 100.0 if i in known_set else 0.0
        for atom in res:
            atom.set_bfactor(bval)

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_pdb))
    print(f" PDB: {output_pdb}")


def plot_rank_distribution(probs, known_indices,
                           output_file='rank_distribution.png'):
    L = len(probs)
    rank, _ = compute_ranks(probs)
    true_ranks = rank[known_indices] / L

    plt.figure(figsize=(8, 5))
    sns.histplot(true_ranks, bins=25, kde=True, color='steelblue',
                 edgecolor='white', alpha=0.7, linewidth=0.5)

    mean_val = np.mean(true_ranks)
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                label=f'Mean = {mean_val:.3f}')

    median_val = np.median(true_ranks)
    plt.axvline(median_val, color='orange', linestyle=':', linewidth=2,
                label=f'Median = {median_val:.3f}')

    plt.xlabel('Rank Percentile', fontsize=13)
    plt.ylabel('Number of Known Residues', fontsize=13)
    plt.title('Ranking of Known Pocket Residues', fontsize=14, pad=15)
    plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    plt.xlim(0, max(0.2, true_ranks.max() * 1.1))
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"save: {output_file}")


def plot_cumulative_recall(probs, known_indices, top_k=TOP_K,
                           output_file='cumulative_recall.png'):

    L = len(probs)
    _, order = compute_ranks(probs)
    known_set = set(known_indices)
    cumulative = []
    hit = 0
    for i, idx in enumerate(order, start=1):
        if idx in known_set:
            hit += 1
        cumulative.append(hit / len(known_set))

    x = np.arange(1, L + 1) / L
    y = np.array(cumulative)

    plt.figure(figsize=(8, 5))
    plt.fill_between(x, y, alpha=0.2, color='darkred')
    plt.plot(x, y, color='darkred', lw=2.5, label='Model')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', lw=1.5,
             label='Random')

    k_frac = top_k / L
    k_recall = y[min(top_k, L) - 1]
    plt.scatter(k_frac, k_recall, color='blue', s=80, zorder=5,
                label=f'Top-{top_k} (recall={k_recall:.2f})')
    plt.axvline(k_frac, color='blue', linestyle=':', alpha=0.5)
    plt.axhline(k_recall, color='blue', linestyle=':', alpha=0.5)

    plt.xlabel('Fraction of Residues Selected', fontsize=13)
    plt.ylabel('Recall of Known Residues', fontsize=13)
    plt.title('Cumulative Recall Curve', fontsize=14, pad=15)
    plt.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"save: {output_file}")


def plot_pr_curve(probs, known_indices, output_file='pr_curve.png'):

    y_true = np.zeros(len(probs))
    y_true[known_indices] = 1
    precision, recall, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, color='blue', lw=2, label=f'AP = {ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"save: {output_file}")


def plot_permutation_rank(null_dist, observed, p_value, z_score,
                          output_file='permutation_rank.png'):

    plt.figure(figsize=(8, 5))
    sns.histplot(null_dist, bins=50, color='grey', alpha=0.6,
                 edgecolor='white', label='Random')
    plt.axvline(observed, color='red', linestyle='--', lw=2,
                label=f'Observed = {observed:.3f}')
    plt.xlabel('Mean Rank Percentile', fontsize=13)
    plt.ylabel('Frequency', fontsize=13)
    plt.title(f'Permutation Test (p = {p_value:.3e}, z = {z_score:.2f})',
              fontsize=14, pad=15)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"save: {output_file}")


def plot_permutation_topk(null_dist, observed, p_value, z_score,
                          output_file='permutation_topk.png'):

    plt.figure(figsize=(8, 5))
    sns.histplot(null_dist, bins=50, color='grey', alpha=0.6,
                 edgecolor='white', label='Random')
    plt.axvline(observed, color='red', linestyle='--', lw=2,
                label=f'Observed = {observed:.3f}')
    plt.xlabel(f'Top-{TOP_K} Recall', fontsize=13)
    plt.ylabel('Frequency', fontsize=13)
    plt.title(f'Permutation Test (p = {p_value:.3e}, z = {z_score:.2f})',
              fontsize=14, pad=15)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"save: {output_file}")


# ================== main ==================
def main():
    alignment = AlignIO.read(MSA_FILE, "clustal")

    chain_lengths = get_chain_lengths(APO_PDB, APO_CHAIN_IDS)
    offsets = {}
    cumulative = 0
    for chain in APO_CHAIN_IDS:
        offsets[chain] = cumulative
        cumulative += chain_lengths.get(chain, 0)
    print(f"9C60 length: {chain_lengths}")
    print(f"off set: {offsets}")


    global_known_indices = []
    for chain_id, res_numbers in COMPLEX_CHAIN_RESIDUES.items():
        seq_indices = residue_numbers_to_sequence_indices(
            COMPLEX_PDB, chain_id, res_numbers)
        query_id = f"8FWQ_{chain_id}"
        target_id = f"9C60_{chain_id}"
        mapped_indices = map_positions(alignment, query_id, target_id,
                                       seq_indices)
        global_mapped = [offsets[chain_id] + idx for idx in mapped_indices]
        global_known_indices.extend(global_mapped)
        print(f"chain {chain_id}: number of residue {len(res_numbers)}, "
              f"Mapping successful {len(mapped_indices)} ")

    global_known_indices = sorted(set(global_known_indices))
    print(f" the number of 9C60 known residues: {len(global_known_indices)}")

    with open(APO_PRED_JSON) as f:
        pred_data = json.load(f)

    if isinstance(pred_data, dict):
        if 'probabilities' in pred_data:
            probs = np.array(pred_data['probabilities'])
        else:
            probs = np.array(list(pred_data.values())[0])
    else:
        probs = np.array(pred_data)


    total_residues = sum(chain_lengths.values())
    if len(probs) != total_residues:
        print(f"Warning：Length of predicted probabilities ({len(probs)}) total the number of residues in the 9C60 standard "
              f"({total_residues}) inconsistency。")


    if len(global_known_indices) == 0:
        print("No Mapping residue。")
        return

    metrics = evaluate(probs, global_known_indices,
                       threshold=PROB_THRESHOLD, top_k=TOP_K)

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")



    obs_rp, p_rp, z_rp, null_rp = permutation_test_rank_percentile(
        probs, global_known_indices, n_perm=N_PERM, seed=RANDOM_SEED)
    print(f"[Mean rank percentile] observed = {obs_rp:.4f}, "
          f"p = {p_rp:.4e}, z = {z_rp:.3f}")

    obs_tk, p_tk, z_tk, null_tk = permutation_test_topk_recall(
        probs, global_known_indices, top_k=TOP_K,
        n_perm=N_PERM, seed=RANDOM_SEED)
    print(f"[Top-{TOP_K} recall] observed = {obs_tk:.4f}, "
          f"p = {p_tk:.4e}, z = {z_tk:.3f}")


    known_pdb = OUTPUT_DIR / "9C60_known_site.pdb"
    write_known_site_pdb(APO_PDB, global_known_indices, known_pdb)

    plot_rank_distribution(
        probs, global_known_indices,
        output_file=str(OUTPUT_DIR / "rank_distribution.png"))
    plot_cumulative_recall(
        probs, global_known_indices, top_k=TOP_K,
        output_file=str(OUTPUT_DIR / "cumulative_recall.png"))
    plot_pr_curve(
        probs, global_known_indices,
        output_file=str(OUTPUT_DIR / "pr_curve.png"))
    plot_permutation_rank(
        null_rp, obs_rp, p_rp, z_rp,
        output_file=str(OUTPUT_DIR / "permutation_rank.png"))
    plot_permutation_topk(
        null_tk, obs_tk, p_tk, z_tk,
        output_file=str(OUTPUT_DIR / "permutation_topk.png"))


    results = {
        'metrics': metrics,
        'permutation_rank_percentile': {
            'observed': float(obs_rp),
            'p_value': float(p_rp),
            'z_score': float(z_rp)
        },
        'permutation_topk_recall': {
            'observed': float(obs_tk),
            'p_value': float(p_tk),
            'z_score': float(z_tk),
            'top_k': TOP_K
        }
    }
    with open(OUTPUT_DIR / "evaluation_summary.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\resluts save:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
