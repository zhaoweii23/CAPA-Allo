#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化版特征提取（ESM + SaProt + GVP + 口袋理化/空间/电子云/氨基酸特征）
输入：蛋白 PDB 文件 + 口袋 PDB 文件
输出：单个 .pkl 文件，包含所有训练/预测所需字段
"""

import pickle
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import dgl
from Bio.PDB import PDBParser, NeighborSearch, Selection
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from gvp import data as gvp_data
from esm import pretrained
from transformers import AutoModel, EsmTokenizer
from torch_scatter import scatter_min
import math
import time
import sys
from tqdm import tqdm
from scipy.spatial import ConvexHull
from scipy.stats import entropy
import networkx as nx

warnings.filterwarnings('ignore', category=PDBConstructionWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==================== 全局配置 ====================
class GlobalConfig:
    # 数据路径
    PROTEIN_PDB_DIR = "data/raw/extracted_chains/"
    ALLOSTERIC_POCKET_DIR = "data/raw/Allosteric_Pocket/"
    FEATURE_PKL_DIR = "data/processed/features_allosteric_pkl_saprot/"

    # 设备
    USE_GPU = torch.cuda.is_available()
    GPU_DEVICE = torch.device('cuda:4' if USE_GPU else 'cpu')

    # 特征开关
    USE_ESM = True
    USE_SAPROT = True

    # ESM 参数
    ESM_MODEL_NAME = "esm2_t33_650M_UR50D"
    ESM_MAX_LEN = 1024
    ESM_SEGMENT_STRIDE = 1024

    # SaProt 参数
    SAPROT_MODEL_PATH = "SaProt_650M_AF2"   # 请确认实际路径

    # 口袋邻域半径
    NEIGHBOR_DISTANCE = 10.0

    # GVP 内坐标截断半径
    GVP_CUTOFF = 8.0

    @classmethod
    def create_directories(cls):
        Path(cls.FEATURE_PKL_DIR).mkdir(parents=True, exist_ok=True)
        Path("data/processed/").mkdir(parents=True, exist_ok=True)


# ==================== 氨基酸映射 ====================
three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
    'TRP': 'W', 'TYR': 'Y'
}


def get_aligned_residues_and_coords(structure):
    """提取排序后的残基、序列和 CA 坐标数组"""
    residues = []
    model = structure[0]
    for chain in model:
        for residue in chain:
            hetero, resid, insertion = residue.full_id[-1]
            if hetero == ' ' and residue.resname in three_to_one:
                if 'CA' in residue:
                    residues.append(residue)
    residues.sort(key=lambda r: (r.get_parent().id, r.id[1], r.id[2]))
    sequence = ''.join(three_to_one[r.resname] for r in residues)
    ca_coords = np.array([r['CA'].coord for r in residues])
    return residues, sequence, ca_coords


def get_residues_from_pocket_pdb(pocket_pdb: str) -> List:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pocket', pocket_pdb)
    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ' and residue.resname in three_to_one:
                    if all(atom in residue for atom in ['N', 'CA', 'C', 'O']):
                        residues.append(residue)
    return residues


def align_pocket_to_residues(pocket_residues, full_residues):
    """返回口袋残基在全蛋白残基列表中的索引"""
    pocket_ids = [(r.get_parent().id, r.id[1], r.id[2]) for r in pocket_residues]
    indices = []
    for pid in pocket_ids:
        for idx, fr in enumerate(full_residues):
            if (fr.get_parent().id, fr.id[1], fr.id[2]) == pid:
                indices.append(idx)
                break
        else:
            logger.warning(f"口袋残基 {pid} 未在全蛋白中找到")
    return indices


def compute_pocket_neighbors(pocket_indices, full_residues, threshold=10.0):
    """根据口袋中心计算邻域残基索引及各残基到口袋中心的距离"""
    ca_coords = np.array([r['CA'].coord for r in full_residues])
    pocket_coords = ca_coords[pocket_indices]
    if len(pocket_coords) == 0:
        return [], [float('inf')] * len(ca_coords)
    pocket_center = np.mean(pocket_coords, axis=0)
    distances = np.linalg.norm(ca_coords - pocket_center, axis=1)
    neighbor_indices = np.where(distances <= threshold)[0].tolist()
    return neighbor_indices, distances.tolist()


def seq_to_high_conf_struc_seq(aa_sequence: str) -> str:
    """将氨基酸序列转换为高置信度结构序列（每个残基：aa+自身折叠字母）"""
    return "".join(f"{aa}{aa}" for aa in aa_sequence)


def find_corresponding_protein(pocket_file: str, protein_dir: str) -> Optional[str]:
    """根据口袋文件名查找对应的完整蛋白 PDB"""
    base_name = Path(pocket_file).stem
    pdb_id = base_name.split('_')[0]
    protein_dir_path = Path(protein_dir)
    candidates = list(protein_dir_path.glob(f"*{pdb_id}*.pdb"))
    return str(candidates[0]) if candidates else None


# ==================== ESM 特征提取器 ====================
class ESM2FeatureExtractor:
    _instance = None
    _initialized = False

    def __new__(cls, config=None):
        if cls._instance is None:
            cls._instance = super(ESM2FeatureExtractor, cls).__new__(cls)
        return cls._instance

    def __init__(self, config=None):
        if not self._initialized:
            self.config = config or GlobalConfig()
            self.device = self.config.GPU_DEVICE
            logger.info(f"初始化 ESM 模型: {self.config.ESM_MODEL_NAME} 设备: {self.device}")
            self.esm_model, self.alphabet = pretrained.load_model_and_alphabet(self.config.ESM_MODEL_NAME)
            self.esm_model = self.esm_model.to(self.device)
            self.esm_model.eval()
            self.batch_converter = self.alphabet.get_batch_converter()
            self._initialized = True

    def extract_esm_token_features(self, sequence: str) -> np.ndarray:
        if not sequence:
            return np.zeros((0, 1280), dtype=np.float32)
        data = [("protein", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
            embeddings = results["representations"][33][0]
        return embeddings[1:len(sequence)+1].cpu().numpy()


# ==================== SaProt 特征提取器 ====================
class SaProtFeatureExtractor:
    _instance = None
    _initialized = False

    def __new__(cls, config=None):
        if cls._instance is None:
            cls._instance = super(SaProtFeatureExtractor, cls).__new__(cls)
        return cls._instance

    def __init__(self, config=None):
        if not self._initialized:
            self.config = config or GlobalConfig()
            self.device = self.config.GPU_DEVICE
            logger.info(f"初始化 SaProt 模型: {self.config.SAPROT_MODEL_PATH} 设备: {self.device}")
            self.tokenizer = EsmTokenizer.from_pretrained(self.config.SAPROT_MODEL_PATH)
            self.model = AutoModel.from_pretrained(self.config.SAPROT_MODEL_PATH)
            self.model = self.model.to(self.device)
            self.model.eval()
            self._initialized = True

    def extract_saprot_token_features(self, struc_seq: str) -> np.ndarray:
        if not struc_seq:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        inputs = self.tokenizer(struc_seq, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0, 1:-1, :]  # (T, D)
        # 取偶数索引作为残基嵌入
        residue_emb = hidden[0::2, :]
        return residue_emb.cpu().numpy()


# ==================== GVP 内坐标计算 ====================
def generate_inner_coor(pos, atom_feats, edge_index, cutoff=8.0):
    """
    计算边上的内坐标特征：dist, theta, phi, tau
    """
    num_nodes = atom_feats.size(0)
    j, i = edge_index
    vecs = pos[j] - pos[i]
    dist = vecs.norm(dim=-1)

    # 第一个最近邻居
    _, argmin0 = scatter_min(dist, i, dim_size=num_nodes)
    argmin0[argmin0 >= len(i)] = 0
    n0 = j[argmin0]
    add = torch.zeros_like(dist).to(dist.device)
    add[argmin0] = cutoff
    dist1 = dist + add
    _, argmin1 = scatter_min(dist1, i, dim_size=num_nodes)
    argmin1[argmin1 >= len(i)] = 0
    n1 = j[argmin1]

    # 发送端最近邻居
    _, argmin0_j = scatter_min(dist, j, dim_size=num_nodes)
    argmin0_j[argmin0_j >= len(j)] = 0
    n0_j = i[argmin0_j]
    add_j = torch.zeros_like(dist).to(dist.device)
    add_j[argmin0_j] = cutoff
    dist1_j = dist + add_j
    _, argmin1_j = scatter_min(dist1_j, j, dim_size=num_nodes)
    argmin1_j[argmin1_j >= len(j)] = 0
    n1_j = i[argmin1_j]

    n0 = n0[i]; n1 = n1[i]; n0_j = n0_j[j]; n1_j = n1_j[j]

    # 参考点选择
    mask_iref = n0 == j
    iref = torch.clone(n0)
    iref[mask_iref] = n1[mask_iref]
    idx_iref = argmin0[i]
    idx_iref[mask_iref] = argmin1[i][mask_iref]

    mask_jref = n0_j == i
    jref = torch.clone(n0_j)
    jref[mask_jref] = n1_j[mask_jref]
    idx_jref = argmin0_j[j]
    idx_jref[mask_jref] = argmin1_j[j][mask_jref]

    pos_ji = vecs
    pos_in0 = vecs[argmin0][i]
    pos_in1 = vecs[argmin1][i]
    pos_iref = vecs[idx_iref]
    pos_jref_j = vecs[idx_jref]

    # 计算角度特征
    a = ((-pos_ji) * pos_in0).sum(dim=-1)
    b = torch.cross(-pos_ji, pos_in0, dim=-1).norm(dim=-1)
    theta = torch.atan2(b, a)
    theta[theta < 0] += math.pi

    dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
    plane1 = torch.cross(-pos_ji, pos_in0, dim=-1)
    plane2 = torch.cross(-pos_ji, pos_in1, dim=-1)
    a = (plane1 * plane2).sum(dim=-1)
    b = (torch.cross(plane1, plane2, dim=-1) * pos_ji).sum(dim=-1) / dist_ji
    phi = torch.atan2(b, a)
    phi[phi < 0] += math.pi

    plane1 = torch.cross(pos_ji, pos_jref_j, dim=-1)
    plane2 = torch.cross(pos_ji, pos_iref, dim=-1)
    a = (plane1 * plane2).sum(dim=-1)
    b = (torch.cross(plane1, plane2, dim=-1) * pos_ji).sum(dim=-1) / dist_ji
    tau = torch.atan2(b, a)
    tau[tau < 0] += math.pi

    return dist, theta, phi, tau


# ==================== 口袋几何/空间特征提取器 ====================
class ProteinSpatialFeatureExtractor:
    def __init__(self):
        self.pdb_parser = PDBParser(QUIET=True)

    def extract_spatial_features(self, pocket_residues: List, full_protein_structure=None) -> Dict:
        try:
            features = {}
            features.update(self._extract_geometric_features(pocket_residues))
            if full_protein_structure is not None:
                features.update(self._extract_network_features(full_protein_structure, pocket_residues))
            features.update(self._extract_spatial_distribution_features(pocket_residues))
            features.update(self._extract_allosteric_signature_features(pocket_residues))
            return features
        except Exception as e:
            logger.error(f"提取空间特征失败: {e}")
            return self._get_default_geometric_features()

    def _extract_geometric_features(self, residues: List) -> Dict:
        if len(residues) < 3:
            return self._get_default_geometric_features()
        ca_coords = [res['CA'].coord for res in residues if 'CA' in res]
        if len(ca_coords) < 3:
            return self._get_default_geometric_features()
        coords_array = np.array(ca_coords)
        try:
            hull = ConvexHull(coords_array)
            volume = hull.volume
            surface_area = hull.area
        except:
            volume = len(residues) * 100
            surface_area = len(residues) * 60
        center = np.mean(coords_array, axis=0)
        distances = np.linalg.norm(coords_array - center, axis=1)
        compactness = np.mean(distances)
        depth = np.max(distances)
        shape_features = self._calculate_shape_descriptors(coords_array)
        return {
            'pocket_volume': volume,
            'pocket_surface_area': surface_area,
            'pocket_compactness': compactness,
            'pocket_depth': depth,
            'residue_count': len(residues),
            **shape_features
        }

    def _calculate_shape_descriptors(self, coords: np.ndarray) -> Dict:
        center = np.mean(coords, axis=0)
        coords_centered = coords - center
        covariance = np.cov(coords_centered.T)
        eigenvalues = np.linalg.eigvals(covariance).real
        eigenvalues_sorted = np.sort(eigenvalues)[::-1]
        if len(eigenvalues_sorted) >= 3:
            asphericity = eigenvalues_sorted[0] - 0.5 * (eigenvalues_sorted[1] + eigenvalues_sorted[2])
            acylindricity = eigenvalues_sorted[1] - eigenvalues_sorted[2]
            relative_shape_anisotropy = (asphericity**2 + 0.75 * acylindricity**2) / np.sum(eigenvalues_sorted)**2
        else:
            asphericity = acylindricity = relative_shape_anisotropy = 0.0
        eigenvalue_ratio_1_2 = eigenvalues_sorted[0] / eigenvalues_sorted[1] if len(eigenvalues_sorted) > 1 and eigenvalues_sorted[1] > 0 else 0.0
        eigenvalue_ratio_2_3 = eigenvalues_sorted[1] / eigenvalues_sorted[2] if len(eigenvalues_sorted) > 2 and eigenvalues_sorted[2] > 0 else 0.0
        return {
            'asphericity': asphericity,
            'acylindricity': acylindricity,
            'shape_anisotropy': relative_shape_anisotropy,
            'eigenvalue_ratio_1_2': eigenvalue_ratio_1_2,
            'eigenvalue_ratio_2_3': eigenvalue_ratio_2_3
        }

    def _extract_spatial_distribution_features(self, residues: List) -> Dict:
        if len(residues) < 3:
            return self._get_default_spatial_features()
        ca_coords = [res['CA'].coord for res in residues if 'CA' in res]
        if len(ca_coords) < 3:
            return self._get_default_spatial_features()
        coords_array = np.array(ca_coords)
        center = np.mean(coords_array, axis=0)
        distances_to_center = np.linalg.norm(coords_array - center, axis=1)
        spatial_clustering = np.std(distances_to_center)
        radial_features = self._calculate_radial_distribution(coords_array, center)
        angular_features = self._calculate_angular_distribution(coords_array, center)
        hull_volume = self._calculate_convex_hull_volume(coords_array)
        density = len(residues) / hull_volume if hull_volume > 0 else 0
        return {
            'spatial_clustering': spatial_clustering,
            'residue_density': density,
            **radial_features,
            **angular_features
        }

    def _calculate_radial_distribution(self, coords: np.ndarray, center: np.ndarray) -> Dict:
        vectors = coords - center
        distances = np.linalg.norm(vectors, axis=1)
        valid_distances = distances[distances > 1e-6]
        if len(valid_distances) == 0:
            return {'radial_std': 0, 'radial_skewness': 0, 'radial_kurtosis': 0, 'radial_asymmetry': 0}
        radial_std = np.std(valid_distances)
        radial_skew = self._calculate_skewness(valid_distances)
        radial_kurtosis = self._calculate_kurtosis(valid_distances)
        unit_vectors = vectors / distances[:, np.newaxis]
        unit_vectors = np.where(distances[:, np.newaxis] > 1e-6, unit_vectors, np.zeros_like(unit_vectors))
        covariance = np.cov(unit_vectors.T)
        eigenvalues = np.linalg.eigvals(covariance).real
        if len(eigenvalues) > 0 and np.mean(eigenvalues) > 1e-6:
            radial_asymmetry = np.std(eigenvalues) / np.mean(eigenvalues)
        else:
            radial_asymmetry = 0.0
        return {
            'radial_std': radial_std,
            'radial_skewness': radial_skew,
            'radial_kurtosis': radial_kurtosis,
            'radial_asymmetry': radial_asymmetry
        }

    def _calculate_angular_distribution(self, coords: np.ndarray, center: np.ndarray) -> Dict:
        vectors = coords - center
        distances = np.linalg.norm(vectors, axis=1)
        valid_mask = distances > 1e-6
        if np.sum(valid_mask) < 2:
            return {'angular_mean': 0, 'angular_std': 0, 'angular_entropy': 0}
        unit_vectors = vectors / distances[:, np.newaxis]
        unit_vectors = unit_vectors[valid_mask]
        angles = []
        n = len(unit_vectors)
        for i in range(n):
            for j in range(i+1, n):
                dot = np.dot(unit_vectors[i], unit_vectors[j])
                dot = np.clip(dot, -1.0, 1.0)
                angles.append(np.arccos(dot))
        if angles:
            angles_array = np.array(angles)
            hist, _ = np.histogram(angles_array, bins=10, density=True)
            hist = hist[hist > 0]
            angular_entropy = entropy(hist) if len(hist) > 0 else 0
            return {
                'angular_mean': np.mean(angles_array),
                'angular_std': np.std(angles_array),
                'angular_entropy': angular_entropy
            }
        return {'angular_mean': 0, 'angular_std': 0, 'angular_entropy': 0}

    def _calculate_skewness(self, data: np.ndarray) -> float:
        n = len(data)
        if n < 3:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        n = len(data)
        if n < 4:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3

    def _extract_network_features(self, full_structure, pocket_residues: List) -> Dict:
        G = self._build_residue_network(full_structure)
        pocket_betweenness = []
        pocket_closeness = []
        pocket_degree = []
        for residue in pocket_residues:
            res_id = self._get_residue_id(residue)
            if G.has_node(res_id):
                betweenness_dict = nx.betweenness_centrality(G)
                closeness_dict = nx.closeness_centrality(G)
                pocket_betweenness.append(betweenness_dict.get(res_id, 0))
                pocket_closeness.append(closeness_dict.get(res_id, 0))
                pocket_degree.append(G.degree(res_id))
        return {
            'mean_betweenness': np.mean(pocket_betweenness) if pocket_betweenness else 0,
            'max_betweenness': np.max(pocket_betweenness) if pocket_betweenness else 0,
            'mean_closeness': np.mean(pocket_closeness) if pocket_closeness else 0,
            'mean_degree': np.mean(pocket_degree) if pocket_degree else 0,
            'network_density': nx.density(G) if len(G) > 0 else 0
        }

    def _build_residue_network(self, structure) -> nx.Graph:
        G = nx.Graph()
        atoms = list(Selection.unfold_entities(structure, 'A'))
        ns = NeighborSearch(atoms)
        pairs = ns.search_all(4.5)
        for atom1, atom2 in pairs:
            if atom1 != atom2:
                res1 = atom1.get_parent()
                res2 = atom2.get_parent()
                if res1 != res2:
                    res1_id = self._get_residue_id(res1)
                    res2_id = self._get_residue_id(res2)
                    dist = np.linalg.norm(atom1.coord - atom2.coord)
                    if dist > 0.1:
                        weight = 1.0 / dist
                        G.add_edge(res1_id, res2_id, weight=weight)
        return G

    def _get_residue_id(self, residue) -> str:
        chain = residue.get_parent()
        return f"{chain.id}_{residue.id[1]}"

    def _calculate_convex_hull_volume(self, coords: np.ndarray) -> float:
        try:
            hull = ConvexHull(coords)
            return hull.volume
        except:
            return len(coords) * 100

    def _extract_allosteric_signature_features(self, residues: List) -> Dict:
        allosteric_hotspots = ['ARG', 'LYS', 'ASP', 'GLU', 'TYR', 'TRP', 'HIS']
        flexible_residues = ['GLY', 'ALA', 'SER', 'THR']
        rigid_residues = ['PRO', 'TRP', 'TYR', 'PHE']
        total = len(residues)
        flexible_count = sum(1 for res in residues if res.resname in flexible_residues)
        rigid_count = sum(1 for res in residues if res.resname in rigid_residues)
        hotspot_coords = [res['CA'].coord for res in residues if res.resname in allosteric_hotspots and 'CA' in res]
        if len(hotspot_coords) > 1:
            hotspot_coords_array = np.array(hotspot_coords)
            center = np.mean(hotspot_coords_array, axis=0)
            distances = np.linalg.norm(hotspot_coords_array - center, axis=1)
            hotspot_clustering = np.std(distances)
        else:
            hotspot_clustering = 0
        return {
            'flexibility_ratio': flexible_count / total if total > 0 else 0,
            'rigidity_ratio': rigid_count / total if total > 0 else 0,
            'conformational_strain': (flexible_count / total) * (1 - (rigid_count / total)) if total > 0 else 0,
            'hotspot_clustering': hotspot_clustering,
            'allosteric_potential': (len(hotspot_coords) / total) * (1 / (1 + hotspot_clustering)) if total > 0 else 0
        }

    def _get_default_geometric_features(self) -> Dict:
        return {
            'pocket_volume': 0, 'pocket_surface_area': 0, 'pocket_compactness': 0,
            'pocket_depth': 0, 'residue_count': 0, 'asphericity': 0,
            'acylindricity': 0, 'shape_anisotropy': 0,
            'eigenvalue_ratio_1_2': 0, 'eigenvalue_ratio_2_3': 0
        }

    def _get_default_spatial_features(self) -> Dict:
        return {
            'spatial_clustering': 0, 'residue_density': 0,
            'radial_std': 0, 'radial_skewness': 0, 'radial_kurtosis': 0,
            'radial_asymmetry': 0, 'angular_mean': 0, 'angular_std': 0, 'angular_entropy': 0
        }


# ==================== 电子云特征提取器 ====================
class ElectronCloudFeatureExtractor:
    def __init__(self):
        self.electronegativity = {
            'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
            'P': 2.19, 'S': 2.58, 'CL': 3.16, 'BR': 2.96, 'I': 2.66,
            'NA': 0.93, 'MG': 1.31, 'CA': 1.00, 'ZN': 1.65, 'FE': 1.83
        }
        self.polarizability = {
            'H': 0.667, 'C': 1.76, 'N': 1.10, 'O': 0.802, 'F': 0.557,
            'P': 3.63, 'S': 2.90, 'CL': 2.18, 'BR': 3.05, 'I': 4.70,
            'NA': 23.6, 'MG': 10.6, 'CA': 22.8, 'ZN': 7.10, 'FE': 8.40
        }
        self.pi_system_residues = ['PHE', 'TYR', 'TRP', 'HIS']

    def extract_electron_cloud_features(self, pocket_residues: List) -> Dict:
        try:
            features = {}
            features.update(self._extract_protein_electron_features(pocket_residues))
            features.update(self._extract_electrostatic_features(pocket_residues))
            features.update(self._extract_aromatic_system_features(pocket_residues))
            features.update(self._extract_hbond_network_features(pocket_residues))
            return features
        except Exception as e:
            logger.error(f"提取电子云特征失败: {e}")
            return self._get_default_electron_features()

    def _extract_protein_electron_features(self, residues: List) -> Dict:
        if not residues:
            return self._get_default_electron_features()
        electron_densities = []
        electronegativities = []
        polarizabilities = []
        for res in residues:
            for atom in res.get_atoms():
                elem = atom.element.strip()
                if elem in self.electronegativity:
                    electronegativities.append(self.electronegativity[elem])
                    polarizabilities.append(self.polarizability.get(elem, 1.0))
                    electron_densities.append(self._get_atomic_number(elem))
        if electron_densities:
            mean_ed = np.mean(electron_densities)
            std_ed = np.std(electron_densities)
        else:
            mean_ed = std_ed = 0.0
        if electronegativities:
            mean_en = np.mean(electronegativities)
            en_range = np.max(electronegativities) - np.min(electronegativities)
        else:
            mean_en = en_range = 0.0
        if polarizabilities:
            mean_pol = np.mean(polarizabilities)
        else:
            mean_pol = 0.0
        return {
            'mean_electron_density': mean_ed,
            'std_electron_density': std_ed,
            'mean_electronegativity': mean_en,
            'electronegativity_range': en_range,
            'mean_polarizability': mean_pol
        }

    def _get_atomic_number(self, element: str) -> int:
        table = {'H':1,'C':6,'N':7,'O':8,'F':9,'P':15,'S':16,'CL':17,'BR':35,'I':53}
        return table.get(element, 6)

    def _extract_electrostatic_features(self, residues: List) -> Dict:
        if not residues:
            return {'esp_surface_variance': 0, 'esp_surface_range': 0, 'electrostatic_asymmetry': 0}
        charge_map = {'ASP':-1,'GLU':-1,'ARG':1,'LYS':1,'HIS':0.5}
        charges = [charge_map.get(res.resname, 0) for res in residues]
        if charges:
            variance = np.var(charges)
            range_val = np.max(charges) - np.min(charges)
            pos = [c for c in charges if c>0]
            neg = [c for c in charges if c<0]
            asymmetry = (sum(pos) - abs(sum(neg))) / (sum(pos)+abs(sum(neg))+1e-6)
        else:
            variance = range_val = asymmetry = 0
        return {
            'esp_surface_variance': variance,
            'esp_surface_range': range_val,
            'electrostatic_asymmetry': asymmetry
        }

    def _extract_aromatic_system_features(self, residues: List) -> Dict:
        aromatic_residues = [r for r in residues if r.resname in self.pi_system_residues]
        if not aromatic_residues:
            return {'aromatic_system_count':0, 'pi_electron_density':0,
                    'aromatic_ring_alignment':0, 'pi_pi_interaction_potential':0}
        pi_electron_density = sum({'PHE':6,'TYR':6,'TRP':10,'HIS':6}.get(r.resname,0) for r in aromatic_residues)
        return {
            'aromatic_system_count': len(aromatic_residues),
            'pi_electron_density': pi_electron_density,
            'aromatic_ring_alignment': 0,
            'pi_pi_interaction_potential': 0
        }

    def _extract_hbond_network_features(self, residues: List) -> Dict:
        donor_residues = {'ARG','LYS','ASN','GLN','SER','THR','TYR','TRP'}
        acceptor_residues = {'ASP','GLU','ASN','GLN','SER','THR','TYR'}
        donor_count = sum(1 for r in residues if r.resname in donor_residues)
        acceptor_count = sum(1 for r in residues if r.resname in acceptor_residues)
        return {
            'hbond_donor_count': donor_count,
            'hbond_acceptor_count': acceptor_count,
            'hbond_network_density': 0,
            'hbond_clustering_coefficient': 0,
            'hbond_network_diameter': 0
        }

    def _get_default_electron_features(self) -> Dict:
        return {
            'mean_electron_density':0,'std_electron_density':0,
            'mean_electronegativity':0,'electronegativity_range':0,
            'mean_polarizability':0,'esp_surface_variance':0,
            'esp_surface_range':0,'electrostatic_asymmetry':0,
            'aromatic_system_count':0,'pi_electron_density':0,
            'aromatic_ring_alignment':0,'pi_pi_interaction_potential':0,
            'hbond_donor_count':0,'hbond_acceptor_count':0,
            'hbond_network_density':0,'hbond_clustering_coefficient':0,
            'hbond_network_diameter':0
        }


# ==================== 氨基酸组成特征提取器 ====================
class AminoAcidFeatureExtractor:
    def __init__(self):
        self.aa_properties = {
            'ALA': {'hydrophobic':1,'polar':0,'charged':0,'small':1,'aromatic':0,'volume':88.6},
            'CYS': {'hydrophobic':1,'polar':1,'charged':0,'small':1,'aromatic':0,'volume':108.5},
            'ASP': {'hydrophobic':0,'polar':1,'charged':-1,'small':1,'aromatic':0,'volume':111.1},
            'GLU': {'hydrophobic':0,'polar':1,'charged':-1,'small':0,'aromatic':0,'volume':138.4},
            'PHE': {'hydrophobic':1,'polar':0,'charged':0,'small':0,'aromatic':1,'volume':189.9},
            'GLY': {'hydrophobic':0,'polar':0,'charged':0,'small':1,'aromatic':0,'volume':60.1},
            'HIS': {'hydrophobic':0,'polar':1,'charged':0.5,'small':0,'aromatic':1,'volume':153.2},
            'ILE': {'hydrophobic':1,'polar':0,'charged':0,'small':0,'aromatic':0,'volume':166.7},
            'LYS': {'hydrophobic':0,'polar':1,'charged':1,'small':0,'aromatic':0,'volume':168.6},
            'LEU': {'hydrophobic':1,'polar':0,'charged':0,'small':0,'aromatic':0,'volume':166.7},
            'MET': {'hydrophobic':1,'polar':0,'charged':0,'small':0,'aromatic':0,'volume':162.9},
            'ASN': {'hydrophobic':0,'polar':1,'charged':0,'small':1,'aromatic':0,'volume':114.1},
            'PRO': {'hydrophobic':1,'polar':0,'charged':0,'small':1,'aromatic':0,'volume':112.7},
            'GLN': {'hydrophobic':0,'polar':1,'charged':0,'small':0,'aromatic':0,'volume':143.8},
            'ARG': {'hydrophobic':0,'polar':1,'charged':1,'small':0,'aromatic':0,'volume':173.4},
            'SER': {'hydrophobic':0,'polar':1,'charged':0,'small':1,'aromatic':0,'volume':89.0},
            'THR': {'hydrophobic':0,'polar':1,'charged':0,'small':1,'aromatic':0,'volume':116.1},
            'VAL': {'hydrophobic':1,'polar':0,'charged':0,'small':1,'aromatic':0,'volume':140.0},
            'TRP': {'hydrophobic':1,'polar':1,'charged':0,'small':0,'aromatic':1,'volume':227.8},
            'TYR': {'hydrophobic':1,'polar':1,'charged':0,'small':0,'aromatic':1,'volume':193.6}
        }
        self.allosteric_hotspots = ['ARG', 'LYS', 'ASP', 'GLU', 'TYR', 'TRP', 'HIS']
        self.pi_system_residues = ['PHE', 'TYR', 'TRP', 'HIS']
        self.flexible_residues = ['GLY', 'ALA', 'SER', 'THR']
        self.rigid_residues = ['PRO', 'TRP', 'TYR', 'PHE']

    def extract_amino_acid_features(self, residues: List) -> Dict:
        if not residues:
            return self._get_default_features()
        property_counts = {
            'hydrophobic':0, 'polar':0, 'charged_positive':0,
            'charged_negative':0, 'small':0, 'aromatic':0
        }
        total_volume = 0
        hotspot_count = 0
        pi_system_count = 0
        flexible_count = 0
        rigid_count = 0
        for res in residues:
            resname = res.resname
            if resname in self.aa_properties:
                props = self.aa_properties[resname]
                property_counts['hydrophobic'] += props['hydrophobic']
                property_counts['polar'] += props['polar']
                if props['charged'] > 0:
                    property_counts['charged_positive'] += 1
                elif props['charged'] < 0:
                    property_counts['charged_negative'] += 1
                property_counts['small'] += props['small']
                property_counts['aromatic'] += props['aromatic']
                total_volume += props['volume']
            if resname in self.allosteric_hotspots:
                hotspot_count += 1
            if resname in self.pi_system_residues:
                pi_system_count += 1
            if resname in self.flexible_residues:
                flexible_count += 1
            if resname in self.rigid_residues:
                rigid_count += 1
        total = len(residues)
        features = {}
        for prop, cnt in property_counts.items():
            features[f'{prop}_ratio'] = cnt / total if total > 0 else 0
        features.update({
            'hotspot_ratio': hotspot_count / total if total > 0 else 0,
            'pi_system_ratio': pi_system_count / total if total > 0 else 0,
            'flexible_ratio': flexible_count / total if total > 0 else 0,
            'rigid_ratio': rigid_count / total if total > 0 else 0,
            'mean_volume': total_volume / total if total > 0 else 0,
            'volume_diversity': np.std([self.aa_properties.get(r.resname, {}).get('volume',0) for r in residues]) if residues else 0
        })
        comp = list(property_counts.values())
        if sum(comp) > 0:
            features['chemical_diversity'] = entropy(comp)
        else:
            features['chemical_diversity'] = 0
        positive_ratio = property_counts['charged_positive'] / total if total > 0 else 0
        negative_ratio = property_counts['charged_negative'] / total if total > 0 else 0
        features['charge_balance'] = positive_ratio - negative_ratio
        features['charge_imbalance'] = abs(positive_ratio - negative_ratio)
        hydrophobic_ratio = property_counts['hydrophobic'] / total if total > 0 else 0
        polar_ratio = property_counts['polar'] / total if total > 0 else 0
        features['hydrophobic_polar_balance'] = hydrophobic_ratio - polar_ratio
        return features

    def _get_default_features(self) -> Dict:
        return {
            'hydrophobic_ratio':0,'polar_ratio':0,'charged_positive_ratio':0,
            'charged_negative_ratio':0,'small_ratio':0,'aromatic_ratio':0,
            'hotspot_ratio':0,'pi_system_ratio':0,'flexible_ratio':0,'rigid_ratio':0,
            'mean_volume':0,'volume_diversity':0,'chemical_diversity':0,
            'charge_balance':0,'charge_imbalance':0,'hydrophobic_polar_balance':0
        }


# ==================== 主特征提取器 ====================
class SimpleProteinFeatureExtractor:
    def __init__(self, config=None):
        self.config = config or GlobalConfig()
        self.device = self.config.GPU_DEVICE
        self.esm_extractor = ESM2FeatureExtractor(config) if self.config.USE_ESM else None
        self.saprot_extractor = SaProtFeatureExtractor(config) if self.config.USE_SAPROT else None

    def _get_esm_embedding(self, sequence: str) -> np.ndarray:
        return self.esm_extractor.extract_esm_token_features(sequence)

    def _get_segmented_esm_embeddings(self, sequence: str) -> List[np.ndarray]:
        max_len = self.config.ESM_MAX_LEN
        stride = self.config.ESM_SEGMENT_STRIDE
        segments = []
        for start in range(0, len(sequence), stride):
            end = min(start + max_len, len(sequence))
            seg_seq = sequence[start:end]
            segments.append(self._get_esm_embedding(seg_seq))
            if end == len(sequence):
                break
        return segments

    def _get_segmented_saprot_embeddings(self, struc_seq: str) -> List[np.ndarray]:
        max_tokens = self.config.ESM_MAX_LEN * 2
        if max_tokens % 2 != 0:
            max_tokens += 1
        stride = max_tokens
        segments = []
        for start in range(0, len(struc_seq), stride):
            end = min(start + max_tokens, len(struc_seq))
            if end % 2 != 0:
                end -= 1
            seg = struc_seq[start:end]
            segments.append(self.saprot_extractor.extract_saprot_token_features(seg))
            if end == len(struc_seq):
                break
        return segments

    def _generate_gvp_from_residues(self, residues: List, full_sequence: str, device=None) -> Tuple:
        structure = {'name': "protein", 'seq': full_sequence, 'coords': []}
        for res in residues:
            res_coords = []
            for atom_name in ['N', 'CA', 'C', 'O']:
                if atom_name in res:
                    res_coords.append(list(res[atom_name].coord))
                else:
                    res_coords.append(res['CA'].coord if 'CA' in res else [0,0,0])
            structure['coords'].append(res_coords)
        torch.set_num_threads(1)
        dataset = gvp_data.ProteinGraphDataset([structure])
        protein = dataset[0]
        s_edge = torch.LongTensor(protein.edge_index[0])
        t_edge = torch.LongTensor(protein.edge_index[1])
        graph = dgl.graph((s_edge, t_edge))
        graph.ndata['h'] = torch.FloatTensor(protein.node_s)
        if device is not None:
            protein.x = protein.x.to(device)
            protein.node_s = protein.node_s.to(device)
            protein.node_v = protein.node_v.to(device)
            protein.edge_index = protein.edge_index.to(device)
            graph = graph.to(device)
        return protein, graph, structure['coords']

    def extract_features(self, pocket_residues: List, full_structure,
                         full_sequence: str, full_residues: List,
                         protein_file: Optional[str] = None) -> Dict:
        features = {}
        seq_len = len(full_sequence)

        # 1. ESM 特征
        if self.config.USE_ESM:
            if seq_len <= self.config.ESM_MAX_LEN:
                features['protein_esm'] = self._get_esm_embedding(full_sequence)
                features['protein_esm_segments'] = None
            else:
                features['protein_esm'] = None
                features['protein_esm_segments'] = self._get_segmented_esm_embeddings(full_sequence)

        # 2. SaProt 特征
        if self.config.USE_SAPROT:
            struc_seq = seq_to_high_conf_struc_seq(full_sequence)
            max_tokens = self.config.ESM_MAX_LEN * 2
            if len(struc_seq) <= max_tokens:
                features['protein_saprot'] = self.saprot_extractor.extract_saprot_token_features(struc_seq)
                features['protein_saprot_segments'] = None
            else:
                features['protein_saprot'] = None
                features['protein_saprot_segments'] = self._get_segmented_saprot_embeddings(struc_seq)

        # 3. 全长 GVP 特征
        full_protein_gvp, full_graph, full_coords = self._generate_gvp_from_residues(
            full_residues, full_sequence, device=self.device
        )
        dist, theta, phi, tau = generate_inner_coor(
            full_protein_gvp.x, full_protein_gvp.node_s, full_protein_gvp.edge_index,
            cutoff=self.config.GVP_CUTOFF
        )

        # 4. 口袋残基索引与邻域信息
        pocket_indices = align_pocket_to_residues(pocket_residues, full_residues)
        neighbor_indices, distances = compute_pocket_neighbors(
            pocket_indices, full_residues, self.config.NEIGHBOR_DISTANCE
        )

        # 5. 口袋空间/几何特征
        spatial_extractor = ProteinSpatialFeatureExtractor()
        spatial_features = spatial_extractor.extract_spatial_features(
            pocket_residues, full_protein_structure=full_structure
        )

        # 6. 电子云特征
        electron_extractor = ElectronCloudFeatureExtractor()
        electron_features = electron_extractor.extract_electron_cloud_features(pocket_residues)

        # 7. 氨基酸组成特征
        aa_extractor = AminoAcidFeatureExtractor()
        aa_features = aa_extractor.extract_amino_acid_features(pocket_residues)

        # 汇总所有特征
        features.update({
            'pdb_id': None,
            'protein_seq': full_sequence,
            'full_protein_gvp': full_protein_gvp,
            'full_coords': full_coords,
            'pocket_indices': pocket_indices,
            'pocket_neighbor_indices': neighbor_indices,
            'distances_to_pocket': distances,
            'full_graph': full_graph,
            'full_dist': dist,
            'full_theta': theta,
            'full_phi': phi,
            'full_tau': tau,
            'spatial_features': spatial_features,
            'electron_features': electron_features,
            'aa_features': aa_features
        })
        return features

    def extract_and_save(self, pocket_file: Path, protein_file: Optional[str] = None) -> Optional[Dict]:
        pdb_id = pocket_file.stem
        pkl_path = Path(self.config.FEATURE_PKL_DIR) / f"{pdb_id}_features.pkl"
        if pkl_path.exists():
            logger.info(f"跳过已处理: {pdb_id}")
            return None

        pocket_residues = get_residues_from_pocket_pdb(str(pocket_file))
        if len(pocket_residues) < 3:
            logger.warning(f"口袋残基数不足: {pocket_file}")
            return None

        if protein_file is None:
            protein_file = find_corresponding_protein(str(pocket_file), self.config.PROTEIN_PDB_DIR)
        if protein_file is None:
            logger.error(f"未找到对应蛋白文件: {pocket_file}")
            return None

        parser = PDBParser(QUIET=True)
        full_structure = parser.get_structure('protein', protein_file)
        full_residues, full_sequence, _ = get_aligned_residues_and_coords(full_structure)
        if not full_sequence:
            logger.error(f"蛋白序列为空: {protein_file}")
            return None

        try:
            features = self.extract_features(pocket_residues, full_structure,
                                             full_sequence, full_residues,
                                             protein_file=protein_file)
        except Exception as e:
            logger.error(f"处理 {pocket_file} 失败: {e}")
            return None

        features['pdb_id'] = pdb_id
        features['site_id'] = f"{pdb_id}_site"

        # 打印长度信息
        log_msg = f"{pdb_id}: seq_len={len(full_sequence)}"
        if features.get('protein_esm') is not None:
            log_msg += f", ESM_len={features['protein_esm'].shape[0]}"
        elif features.get('protein_esm_segments'):
            log_msg += f", ESM_segments={len(features['protein_esm_segments'])}"
        if features.get('protein_saprot') is not None:
            log_msg += f", SaProt_len={features['protein_saprot'].shape[0]}"
        elif features.get('protein_saprot_segments'):
            log_msg += f", SaProt_segments={len(features['protein_saprot_segments'])}"
        log_msg += f", pocket={len(features['pocket_indices'])}"
        logger.info(log_msg)

        with open(pkl_path, 'wb') as f:
            pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"特征保存: {pkl_path}")
        return features


# ==================== 主程序 ====================
def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('feature_extraction_with_saprot_20260522.log'),
            logging.StreamHandler()
        ]
    )
    logger.info("=== 特征提取系统启动（ESM + SaProt + GVP + 口袋理化特征）===")
    start_time = time.time()

    GlobalConfig.create_directories()

    pocket_dir = Path(GlobalConfig.ALLOSTERIC_POCKET_DIR)
    if not pocket_dir.exists():
        logger.error(f"口袋目录不存在: {pocket_dir}")
        sys.exit(1)

    pocket_files = list(pocket_dir.glob("*.pdb"))
    logger.info(f"找到 {len(pocket_files)} 个口袋PDB文件")

    extractor = SimpleProteinFeatureExtractor()
    success_count = 0
    for pf in tqdm(pocket_files, desc="提取特征"):
        result = extractor.extract_and_save(pf)
        if result is not None:
            success_count += 1

    elapsed = time.time() - start_time
    logger.info(f"处理完成: 成功 {success_count} 个，失败 {len(pocket_files)-success_count} 个")
    logger.info(f"总耗时: {elapsed:.2f} 秒")


if __name__ == "__main__":
    main()
