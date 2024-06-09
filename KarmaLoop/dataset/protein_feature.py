#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2022/8/8 14:23
# @author : Xujun Zhang, Tianyue Wang

import math
import numpy as np
import torch
import torch.nn.functional as F
import torch_cluster
import MDAnalysis as mda
from scipy.spatial import distance_matrix
from MDAnalysis.analysis import distances

METAL = ["LI","NA","K","RB","CS","MG","TL","CU","AG","BE","NI","PT","ZN","CO","PD","AG","CR","FE","V","MN","HG",'GA', 
		"CD","YB","CA","SN","PB","EU","SR","SM","BA","RA","AL","IN","TL","Y","LA","CE","PR","ND","GD","TB","DY","ER",
		"TM","LU","HF","ZR","CE","U","PU","TH"] 
three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y', 'UNK': 'X'}
three2idx = {k:v for v, k in enumerate(['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 
										'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 
										'GLU', 'LYS', 'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO',
										'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP', 'UNK'])}
three2self = {v:v for v in ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 
										'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 
										'GLU', 'LYS', 'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO',
										'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP']}
RES_MAX_NATOMS=24

Seq2Hydropathy = {'X': 0, "I":4.5, "V":4.2, "L":3.8, "F":2.8, "C":2.5, "M":1.9, "A":1.8, "W":-0.9, "G":-0.4, "T":-0.7, "S":-0.8, "Y":-1.3, "P":-1.6, "H":-3.2, "N":-3.5, "D":-3.5, "Q":-3.5, "E":-3.5, "K":-3.9, "R":-4.5}
Seq2Volume = {'X': 0, "G":60.1, "A":88.6, "S":89.0, "C":108.5, "D":111.1, "P":112.7, "N":114.1, "T":116.1, "E":138.4, "V":140.0, "Q":143.8, "H":153.2, "M":162.9, "I":166.7, "L":166.7, "K":168.6, "R":173.4, "F":189.9, "Y":193.6, "W":227.8}
Seq2Charge = {**{'R':1, 'K':1, 'D':-1, 'E':-1, 'H':0.1}, **{x:0 for x in 'ABCFGIJLMNOPQSTUVWXYZX'}}
Seq2Polarity = {**{x:1 for x in 'RNDQEHKSTY'}, **{x:0 for x in "ACGILMFPWVX"}}
Seq2Acceptor = {**{x:1 for x in 'DENQHSTY'}, **{x:0 for x in "RKWACGILMFPVX"}}
Seq2Donor = {**{x:1 for x in 'RKWNQHSTY'}, **{x:0 for x in "DEACGILMFPVX"}}

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def np_rbf(D, D_min=0., D_max=20., D_count=16):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = np.linspace(D_min, D_max, D_count)
    # D_mu = D_mu.reshape((1, -1))
    D_sigma = (D_max - D_min) / D_count
    # D_expand = np.expand_dims(D, axis=-1)

    RBF = np.exp(-((D - D_mu) / D_sigma) ** 2)
    return RBF


def obtain_resname(res):
	if res.resname[:2] == "CA":
		resname = "CA"
	elif res.resname[:2] == "FE":
		resname = "FE"
	elif res.resname[:2] == "CU":
		resname = "CU"
	else:
		resname = res.resname.strip()
	
	if resname in METAL:
		return "M"
	else:
		return resname


def obtain_dihediral_angles(res):
    angle_lis = [0, 0, 0, 0]
    for idx, angle in enumerate([res.phi_selection, res.psi_selection, res.omega_selection, res.chi1_selection]):
        try:
            angle_lis[idx] = angle().dihedral.value()
        except:
            continue
    return angle_lis


def obtain_X_atom_pos(res, name='CA'):
	if obtain_resname(res) == "M":
		return res.atoms.positions[0]
	else:
		try:
			pos = res.atoms.select_atoms(f"name {name}").positions[0]
			return pos
		except:  ##some residues loss the CA atoms
			return res.atoms.positions.mean(axis=0)


def obtain_self_dist(res):
	try:
		#xx = res.atoms.select_atoms("not name H*")
		xx = res.atoms
		dists = distances.self_distance_array(xx.positions)
		ca = xx.select_atoms("name CA")
		c = xx.select_atoms("name C")
		n = xx.select_atoms("name N")
		o = xx.select_atoms("name O")
		return [dists.max()*0.1, dists.min()*0.1, distances.dist(ca,o)[-1][0]*0.1, distances.dist(o,n)[-1][0]*0.1, distances.dist(n,c)[-1][0]*0.1]
	except:
		return [0, 0, 0, 0, 0]


def calc_res_features(res):
	return np.array(
			obtain_self_dist(res) +  #5
			obtain_dihediral_angles(res) #4		
			)


def check_connect(res_lis, i, j):
    if abs(i-j) == 1 and res_lis[i].segid == res_lis[j].segid:
        return 1
    else:
        return 0


def positional_embeddings_v1(edge_index,
                                num_embeddings=16,
                                period_range=[2, 1000]):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    d = edge_index[0] - edge_index[1]
    # raw
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    # # new
    # max_relative_feature = 32
    # d = torch.clip(d + max_relative_feature, 0, 2 * max_relative_feature)
    # d_onehot = F.one_hot(d, 2 * max_relative_feature + 1)
    # E = torch.cat((torch.cos(angles), torch.sin(angles), d_onehot), -1)
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E


def calc_dist(res1, res2):
	dist_array = distances.distance_array(res1.atoms.positions,res2.atoms.positions)
	return dist_array


def obatin_edge(res_lis, src, dst):
    dist = calc_dist(res_lis[src], res_lis[dst])
    return dist.min()*0.1, dist.max()*0.1


def get_orientations(X):
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def get_sidechains(n, ca, c):
    c, n = _normalize(c - ca), _normalize(n - ca)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec


def get_protein_feature_mda(pocket_mol, top_k=30):
    with torch.no_grad():
        pure_res_lis, pure_res_key, pure_resname, seq, node_s, X_ca, X_n, X_c = [], [], [], [], [], [], [], []
        for res in pocket_mol.residues:
            try:
                res_name = res.resname.strip()
                res_atoms = res.atoms
                dists = distances.self_distance_array(res_atoms.positions)
                ca = res_atoms.select_atoms("name CA")
                c = res_atoms.select_atoms("name C")
                n = res_atoms.select_atoms("name N")
                o = res_atoms.select_atoms("name O")
                dis_max = dists.max()
                dis_min = dists.min()
                dis_cao = distances.dist(ca,o)[-1][0]
                dis_on = distances.dist(o,n)[-1][0]
                dis_nc = distances.dist(n,c)[-1][0]
                intra_dis = [dis_max*0.1, dis_min*0.1, dis_cao*0.1, dis_on*0.1, dis_nc*0.1] + \
                             np_rbf(dis_max, D_min=0, D_max=20, D_count=16).tolist() + \
                             np_rbf(dis_min, D_min=0, D_max=1.5, D_count=16).tolist() + \
                             np_rbf(dis_cao, D_min=0, D_max=3.5, D_count=16).tolist() + \
                             np_rbf(dis_on, D_min=0, D_max=4.5, D_count=16).tolist() + \
                             np_rbf(dis_nc, D_min=0, D_max=3.5, D_count=16).tolist() 
                seq_name_3 = three2self.get(res_name, 'UNK')
                seq_name_1 = three_to_one.get(seq_name_3, 'X')
                seq.append(three2idx[seq_name_3])
                X_ca.append(ca.positions[0])
                X_n.append(n.positions[0])
                X_c.append(c.positions[0])
                res_feats = [Seq2Hydropathy[seq_name_1], Seq2Volume[seq_name_1] / 100, Seq2Charge[seq_name_1],
                Seq2Polarity[seq_name_1], Seq2Acceptor[seq_name_1], Seq2Donor[seq_name_1]]
                node_s.append(intra_dis+obtain_dihediral_angles(res)+res_feats)
                pure_res_lis.append(res)
                pure_res_key.append(f'{res_atoms.chainIDs[0]}-{res.resid}{res.icode}') 
                pure_resname.append(f'{res.resname}{res.resid}{res.icode}')
            except:
                continue

        # node features
        seq = torch.from_numpy(np.asarray(seq))
        node_s = torch.from_numpy(np.asarray(node_s))
        # edge features
        X_ca = torch.from_numpy(np.asarray(X_ca))	
        X_n = torch.from_numpy(np.asarray(X_n))	
        X_c = torch.from_numpy(np.asarray(X_c))	
        X_center_of_mass = torch.from_numpy(pocket_mol.atoms.center_of_mass(compound='residues'))
        edge_index = torch_cluster.knn_graph(X_ca, k=top_k)
        dis_minmax = torch.from_numpy(np.asarray([obatin_edge(pure_res_lis, src, dst) for src, dst in edge_index.T])).view(edge_index.size(1), 2)
        dis_matx_center = distance_matrix(X_center_of_mass, X_center_of_mass)
        cadist = (torch.pairwise_distance(X_ca[edge_index[0]], X_ca[edge_index[1]]) * 0.1).view(-1,1)
        cedist = (torch.from_numpy(dis_matx_center[edge_index[0,:], edge_index[1,:]]) * 0.1).view(-1,1)
        edge_connect =  torch.from_numpy(np.asarray([check_connect(pure_res_lis, x, y) for x,y in edge_index.T])).view(-1,1)
        positional_embedding = positional_embeddings_v1(edge_index)
        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        edge_s = torch.cat([edge_connect, cadist, cedist, dis_minmax, 
                            _rbf(cadist.view(-1), D_min=0, D_max=2.0, D_count=16, device='cpu'), 
                            _rbf(cedist.view(-1), D_min=0, D_max=2.0, D_count=16, device='cpu'), 
                            _rbf(dis_minmax[:, 0].view(-1), D_min=0, D_max=1.2, D_count=16, device='cpu'),
                            _rbf(dis_minmax[:, 1].view(-1), D_min=0, D_max=8, D_count=16, device='cpu'),
                            positional_embedding], dim=1)
        # vector features
        orientations = get_orientations(X_ca)
        sidechains = get_sidechains(n=X_n, ca=X_ca, c=X_c)
        node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
        edge_v = _normalize(E_vectors).unsqueeze(-2)
        xyz_full = torch.from_numpy(np.asarray([np.concatenate([res.atoms.positions[:RES_MAX_NATOMS, :], np.full((max(RES_MAX_NATOMS-len(res.atoms), 0), 3), np.nan)],axis=0) for res in pure_res_lis]))
        node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,(node_s, node_v, edge_s, edge_v))
        # full edge
        full_edge_s = torch.zeros((edge_index.size(1), 5))  # [0, 0, 0, 0, 0]
                                                              # [s, d, t, f, non-cov]
        full_edge_s[edge_s[:, 0]==1, 0] = 1
        full_edge_s[edge_s[:, 0]==0, 4] = 1
        full_edge_s = torch.cat([full_edge_s, cadist], dim=-1)
        return (X_ca, xyz_full, seq, node_s, node_v, edge_index, edge_s, edge_v, full_edge_s, pure_res_key, pure_resname)        # full edge



