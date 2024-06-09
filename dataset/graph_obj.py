#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2022/3/28 14:08
# @author : Xujun Zhang, Tianyue Wang
import prody
import os
import sys
import MDAnalysis as mda
from functools import partial 
from multiprocessing import Pool
import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RDLogger.DisableLog("rdApp.*")
# dir of current
from utils.fns import load_graph, save_graph
from dataset.protein_feature import get_protein_feature_mda
from dataset.loop_feature import get_loop_feature_v1

print = partial(print, flush=True)
   
class LoopGraphDataset(Dataset):

    def __init__(self, graph_dir, pdb_ids, 
                 geometric_pos_init=False, multi_conformation=False, sigma=1.75):
        self.graph_dir = graph_dir
        os.makedirs(graph_dir, exist_ok=True)
        self.pdb_ids = pdb_ids
        self.sigma = sigma if not multi_conformation else 7
        self.geometric_pos_init = geometric_pos_init

    def generate_graph(self, pocket_dir, protein_dir='', n_job=1, verbose=True):
        self.pocket_dir = pocket_dir
        self.protein_dir = protein_dir
        idxs = range(len(self.pdb_ids))
        if verbose:
            print('### generating graph')
        single_process = partial(self._single_process, return_graph=False, save_file=True)
        # generate graph
        if n_job == 1:
            if verbose:
                idxs = tqdm(idxs)
            for idx in idxs:
                single_process(idx)
        else:
            pool = Pool()
            pool.map(single_process, idxs)
            pool.close()
            pool.join()

    def _single_process(self, idx, return_graph=False, save_file=False):
        pdb_id = self.pdb_ids[idx]
        dst_file = f'{self.graph_dir}/{"_".join(pdb_id.split("_")[:4])}.dgl'
        if os.path.exists(dst_file):
            # pass
            # reload graph
            if return_graph:
                return load_graph(dst_file)
        else:
            # generate graph
            try:
                data = get_loop_graph(pdb_id=pdb_id,
                                    pocket_dir=self.pocket_dir, 
                                    protein_dir=self.protein_dir)
                data.pdb_id = pdb_id
                if save_file:
                    save_graph(dst_file, data)
                if return_graph:
                    return data
            except Exception as e:
                print(f'##################################\n# {pdb_id} error due to {e}\n##################################')
                return None

    def __getitem__(self, idx):
        # load graph
        data = self._single_process(idx=idx, return_graph=True, save_file=False)
        # loop initial position
        if self.geometric_pos_init == 'anchor':
            data['loop'].pos = torch.clamp(torch.randn_like(data['loop'].xyz) * self.sigma, min=-self.sigma, max=self.sigma)
            delta_x = data.anchor_dst_pos - data.anchor_src_pos  # (1, 3)
            anchor_center = [data.anchor_src_pos + t*delta_x for t in np.linspace(0, 1, (data.loop_len)+2)][1:-1]
            for l_idx in range(data.loop_len):
                select_mask = data.atom2loopres == l_idx
                data['loop'].pos[select_mask] -= data['loop'].pos[select_mask].mean(dim=0) - anchor_center[l_idx]
        else:
            data['loop'].pos = torch.randn_like(data['loop'].xyz) * 4
            data['loop'].pos -= data['loop'].pos.mean(axis=0) - data.pocket_center
        return data


    def __len__(self):
        return len(self.pdb_ids)


def get_repeat_node(src_num, dst_num):
    return torch.arange(src_num, dtype=torch.long).repeat(dst_num), \
           torch.as_tensor(np.repeat(np.arange(dst_num), src_num), dtype=torch.long)


def pdb2rdmol(pocket_pdb):
    pocket_atom_mol = Chem.MolFromPDBFile(pocket_pdb, removeHs=False, sanitize=False)
    pocket_atom_mol = Chem.RemoveAllHs(pocket_atom_mol, sanitize=False)
    return pocket_atom_mol


def get_loop_graph(pdb_id, pocket_dir, protein_dir):
    torch.set_num_threads(1)
    pocket_pdb = f'{pocket_dir}/{pdb_id}_pocket_12A.pdb'
    pdbid, chain, res_num_src, res_num_dst = pdb_id.split('_')[:4]
    res_num_src, res_num_dst = int(res_num_src), int(res_num_dst)
    # pocket_center = np.asarray([x_, y_, z_], dtype=np.float32)
    loop_len = res_num_dst - res_num_src + 1
    nearby_resid_src, nearby_resid_dst = res_num_src-1, res_num_dst+1
    # get protein mol
    pocket_mol = mda.Universe(pocket_pdb)
    nearby_res_src = pocket_mol.select_atoms(f'chainid {chain} and (resnum {int(nearby_resid_src)})')
    nearby_res_dst = pocket_mol.select_atoms(f'chainid {chain} and (resnum {int(nearby_resid_dst)})')
    assert len(nearby_res_src) != 0, 'nan in anchor_src_pos'
    assert len(nearby_res_dst) != 0, 'nan in anchor_dst_pos'
    nearby_res_src = nearby_res_src.positions.mean(axis=0)
    nearby_res_dst = nearby_res_dst.positions.mean(axis=0)
    # get pocket mol
    non_loop_mol = pocket_mol.select_atoms(f'not (chainid {chain} and resnum {res_num_src}:{res_num_dst})')
    loop_mol = pocket_mol.select_atoms(f'chainid {chain} and (resnum {res_num_src}:{res_num_dst})')
    loop_res_num = len(loop_mol.residues)
    # assert loop_len == len(loop_mol.residues), f'{pdb_id} loop length error'
    if loop_len != loop_res_num:
        print(f"Warning: For protein {pdb_id}, the number of residues in the loop does not match the count derived from the loop's start and end indices. This discrepancy may be acceptable if the protein functions as an antigen or antibody. Otherwise, a thorough review of this protein is advised.")
        loop_len = loop_res_num
    loop_res = [f'{chain}-{res.resid}{res.icode}' for res in loop_mol.residues]
    pocket_atom_mol = pdb2rdmol(pocket_pdb)
    # get protein mol
    protein_pdb = f'{protein_dir}/{pdbid}.pdb'
    if os.path.exists(protein_pdb):
        protein_atom_mol = pdb2rdmol(protein_pdb)
    else:
        protein_atom_mol = None
    # generate graph
    p_xyz, p_xyz_full, p_seq, p_node_s, p_node_v, p_edge_index, p_edge_s, p_edge_v, p_full_edge_s, p_node_name, p_node_type = get_protein_feature_mda(non_loop_mol)
    loop_name, loop_xyz, pa_node_feature, pa_edge_index, pa_edge_feature, atom2nonloopres, nonloop_mask, loop_edge_mask, loop_edge_index, loop_edge_feature, loop_cov_edge_mask, loop_frag_edge_mask, loop_idx_2_mol_idx, atom2loopres = get_loop_feature_v1(pocket_atom_mol, nonloop_res=p_node_name, loop_res=loop_res)
    # to data
    data = HeteroData()
    # protein residue
    # data.pocket_center = torch.from_numpy(pocket_center).to(torch.float32)
    data.atom2nonloopres = torch.tensor(atom2nonloopres, dtype=torch.long)
    data.atom2loopres = torch.tensor(atom2loopres, dtype=torch.long)
    data.nonloop_mask = torch.tensor(nonloop_mask, dtype=torch.bool)
    data.loop_edge_mask = torch.tensor(loop_edge_mask, dtype=torch.bool)
    data.loop_cov_edge_mask = loop_cov_edge_mask
    data.loop_frag_edge_mask = loop_frag_edge_mask
    data.loop_idx_2_mol_idx = loop_idx_2_mol_idx
    data.mol = pocket_atom_mol
    data.protein_mol = protein_atom_mol
    data.loop_len = loop_len
    data.anchor_src_pos = torch.from_numpy(nearby_res_src).to(torch.float32).view(1, 3)
    data.anchor_dst_pos = torch.from_numpy(nearby_res_dst).to(torch.float32).view(1, 3)
    # data['protein'].node_name = p_node_type
    data['protein'].node_s = p_node_s.to(torch.float32) 
    data['protein'].node_v = p_node_v.to(torch.float32)
    data['protein'].xyz = p_xyz.to(torch.float32) 
    data['protein'].xyz_full = p_xyz_full.to(torch.float32) 
    data['protein'].seq = p_seq.to(torch.int32)
    data['protein', 'p2p', 'protein'].edge_index = p_edge_index.to(torch.long)
    data['protein', 'p2p', 'protein'].edge_s = p_edge_s.to(torch.float32) 
    data['protein', 'p2p', 'protein'].full_edge_s = p_full_edge_s.to(torch.float32) 
    data['protein', 'p2p', 'protein'].edge_v = p_edge_v.to(torch.float32) 
    # protein atom
    data['protein_atom'].node_s = pa_node_feature.to(torch.float32) 
    data['protein_atom', 'pa2pa', 'protein_atom'].edge_index = pa_edge_index.to(torch.long)
    data['protein_atom', 'pa2pa', 'protein_atom'].edge_s = pa_edge_feature.to(torch.float32) 
    # # loop
    # data['loop'].node_name = loop_name
    data['loop'].node_s = torch.zeros((data.nonloop_mask.sum(), 1))
    data['loop'].xyz = loop_xyz.to(torch.float32)
    data['loop', 'l2l', 'loop'].edge_index = loop_edge_index.to(torch.long)
    data['loop', 'l2l', 'loop'].full_edge_s = loop_edge_feature.to(torch.float32)
    # # protein-loop
    data['protein', 'p2l', 'loop'].edge_index = torch.stack(
        get_repeat_node(p_xyz.shape[0], loop_xyz.shape[0]), dim=0)
    return data


if __name__ == '__main__':
    p_path = '/root/surface_pocket'
    pdb_id = '6RW7_A_55_78'
    get_loop_graph(pdb_id, p_path)

