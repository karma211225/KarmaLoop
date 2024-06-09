#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2022/8/8 16:51
# @author : Xujun Zhang, Tianyue Wang

import torch
import random
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
from architecture.GVP_Block import GVP_embedding
from architecture.GraphTransformer_Block import GraghTransformer
from architecture.MDN_Block import MDN_Block
from architecture.EGNN_Block import EGNN
from architecture.Gate_Block import Gate_Block
from torch_scatter import scatter_mean, scatter
from torch_geometric.nn import GraphNorm
from utils.fns import mdn_loss_fn


class KarmaLoop(nn.Module):
    def __init__(self, hierarchical=True):
        super(KarmaLoop, self).__init__()
        self.hierarchical = hierarchical
        # encoders
        self.loop_encoder = GraghTransformer(
            in_channels=89, 
            edge_features=20, 
            num_hidden_channels=128,
            activ_fn=torch.nn.SiLU(),
            transformer_residual=True,
            num_attention_heads=4,
            norm_to_apply='batch',
            dropout_rate=0.15,
            num_layers=6
        )
        self.pro_encoder = GVP_embedding(
            (95, 3), (128, 16), (85, 1), (32, 1), seq_in=True, vector_gate=True) 
        self.gn = GraphNorm(128)
        # pose prediction
        self.egnn_layers = nn.ModuleList( 
            [EGNN(dim_in=128, dim_tmp=128, edge_in=128, edge_out=128, num_head=4, drop_rate=0.15) for i in range(8)]
        )
        self.edge_init_layer = nn.Linear(6, 128)
        self.node_gate_layer = Gate_Block(dim_tmp=128, 
                                        drop_rate=0.15
                                        )
        self.edge_gate_layer = Gate_Block(dim_tmp=128, 
                                        drop_rate=0.15
                                        )
        if hierarchical:
            self.merge_hierarchical = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(p=0.15),
            nn.LeakyReLU(),
            nn.Linear(256, 128)
        )
            self.graph_norm = GraphNorm(in_channels=128)
        # scoring 
        self.mdn_layer = MDN_Block(hidden_dim=128, 
                                         n_gaussians=10, 
                                        dropout_rate=0.10, 
                                        dist_threhold=7.)


    def cal_rmsd(self, pos_ture, pos_pred, batch, if_r=True):
        if if_r:
            return scatter_mean(((pos_pred - pos_ture)**2).sum(dim=-1), batch).sqrt()
        else:
            return scatter_mean(((pos_pred - pos_ture)**2).sum(dim=-1), batch)
    
    
    def encoding(self, data):
        '''
        get loop & protein embeddings
        '''
        # encoder 
        proatom_node_s = self.loop_encoder(data['protein_atom'].node_s.to(torch.float32), data['protein_atom', 'pa2pa', 'protein_atom'].edge_s.to(torch.float32), data['protein_atom', 'pa2pa', 'protein_atom'].edge_index)
        pro_node_s = self.pro_encoder((data['protein']['node_s'], data['protein']['node_v']),
                                                      data[(
                                                          "protein", "p2p", "protein")]["edge_index"],
                                                      (data[("protein", "p2p", "protein")]["edge_s"],
                                                       data[("protein", "p2p", "protein")]["edge_v"]),
                                                      data['protein'].seq)
        loop_node_s = proatom_node_s[data.nonloop_mask]
        proatom_node_s = proatom_node_s[~data.nonloop_mask]
        max_res_num_per_sample = data.atom2nonloopres[data.atom2nonloopres_ptr[1:-1] - 1]
        atom2res = data.atom2nonloopres.clone()
        for idxb, b in enumerate(data.atom2nonloopres_ptr[1:-1]):
            atom2res[b:] += max_res_num_per_sample[idxb] + 1 
        del max_res_num_per_sample
        proatom_node_s = scatter(proatom_node_s, index=atom2res, reduce='sum', dim=0, dim_size=pro_node_s.size(0))
        del atom2res
        proatom_node_s = self.graph_norm(proatom_node_s, data['protein'].batch)
        pro_node_s = torch.cat([pro_node_s, proatom_node_s], dim=-1)
        pro_node_s = self.merge_hierarchical(pro_node_s)
        del proatom_node_s
        return pro_node_s, loop_node_s
    
    def scoring(self, loop_s, loop_pos, pro_s, data, dist_threhold, batch_size, train=True):
        '''
        confidence score
        '''
        pi, sigma, mu, dist, c_batch, atom_types, bond_types = self.mdn_layer(loop_s=loop_s, loop_pos=loop_pos, loop_batch=data['loop'].batch,
                                                               pro_s=pro_s, pro_pos=data['protein'].xyz_full, pro_batch=data['protein'].batch,
                                                               edge_index=data['loop', 'l2l', 'loop'].edge_index[:, data.loop_cov_edge_mask], train=train)
        if train:
            ## mdn aux labels
            aux_r = 0.001
            atom_types_label = torch.argmax(data['protein_atom'].node_s[data.nonloop_mask,:18], dim=1, keepdim=False)
            bond_types_label = torch.argmax(data['protein_atom', 'pa2pa', 'protein_atom'].edge_s[data.loop_edge_mask, :5], dim=1, keepdim=False)
            mdn_score = mdn_loss_fn(pi, sigma, mu, dist)[torch.where(dist <= self.mdn_layer.dist_threhold)[0]].mean().float()+ aux_r*F.cross_entropy(atom_types, atom_types_label) + aux_r*F.cross_entropy(bond_types, bond_types_label)
            del pi, sigma, mu, dist, c_batch, atom_types, bond_types, atom_types_label, bond_types_label
        else:
            mdn_score = self.mdn_layer.calculate_probablity(pi, sigma, mu, dist)
            mdn_score[torch.where(dist > dist_threhold)[0]] = 0.
            mdn_score = scatter(mdn_score, index=c_batch, dim=0, reduce='sum', dim_size=batch_size).float()
        return mdn_score
    
    def loop_modelling(self, pro_node_s, loop_node_s, data, recycle_num=3, train=False):
        '''
        generate loop conformations 
        '''
        # graph norm through interaction graph
        pro_nodes = data['protein'].num_nodes
        node_s = self.gn(torch.cat([pro_node_s, loop_node_s], dim=0), torch.cat([data['protein'].batch, data['loop'].batch], dim=-1))
        data['protein'].node_s, data['loop'].node_s = node_s[:pro_nodes], node_s[pro_nodes:]
        del node_s
        # build interaction graph
        pro_nodes = data['protein'].num_nodes
        batch = torch.cat([data['protein'].batch, data['loop'].batch], dim=-1)
        u = torch.cat([data[("protein", "p2p", "protein")]["edge_index"][0], data[('loop', 'l2l', 'loop')]["edge_index"][0]+pro_nodes, data[('protein', 'p2l', 'loop')]["edge_index"][0], data[('protein', 'p2l', 'loop')]["edge_index"][1]+pro_nodes], dim=-1)
        v = torch.cat([data[("protein", "p2p", "protein")]["edge_index"][1], data[('loop', 'l2l', 'loop')]["edge_index"][1]+pro_nodes, data[('protein', 'p2l', 'loop')]["edge_index"][1]+pro_nodes, data[('protein', 'p2l', 'loop')]["edge_index"][0]], dim=-1)
        edge_index = torch.stack([u, v], dim=0)
        del u, v
        node_s = torch.cat([data['protein'].node_s, data['loop'].node_s], dim=0)
        edge_s = torch.zeros((data[('protein', 'p2l', 'loop')]["edge_index"][0].size(0)*2, 6), device=node_s.device)
        edge_s[:, -1] = -1
        edge_s = torch.cat([data[("protein", "p2p", "protein")].full_edge_s, data['loop', 'l2l', 'loop'].full_edge_s, edge_s], dim=0)
        pos = torch.cat([data['protein'].xyz, data['loop'].pos], dim=0)
        # EGNN
        edge_s = self.edge_init_layer(edge_s)
        pos_mdn_losss = []
        for re_idx in range(recycle_num):
            for l_idx, layer in enumerate(self.egnn_layers):
                node_s, edge_s, pos = layer(node_s, edge_s, edge_index, pos, pro_nodes, batch, update_pos=True)
                pos_mdn_losss.append(self.cal_rmsd(pos_ture=data['loop'].xyz, pos_pred=pos[pro_nodes:], batch=data['loop'].batch, if_r=True))
            node_s = self.node_gate_layer(torch.cat([data['protein'].node_s, data['loop'].node_s], dim=0), node_s)
            edge_s = self.edge_gate_layer(
                            self.edge_init_layer(torch.cat([
                                data[("protein", "p2p", "protein")].full_edge_s,
                                data['loop', 'l2l', 'loop'].full_edge_s,
                                        torch.cat([torch.zeros((data[('protein', 'p2l', 'loop')]["edge_index"][0].size(0)*2, 5), device=node_s.device),
                                        -torch.ones((data[('protein', 'p2l', 'loop')]["edge_index"][0].size(0)*2, 1), device=node_s.device),
                                        ], dim=1)], dim=0)), 
                            edge_s)
        del node_s, edge_s, edge_index, batch 
        if train:    
            count_idx = random.choice(range(recycle_num))
            pos_mdn_losss = torch.stack(pos_mdn_losss, dim=0)
            rmsd_loss = pos_mdn_losss[-1]
            pos_mdn_losss = (pos_mdn_losss[8*count_idx:8*(count_idx+1)].mean(dim=0) + rmsd_loss)
            # print(rmsd_loss)
            return pos[pro_nodes:], data['loop'].xyz, data['loop'].batch, rmsd_loss, pos_mdn_losss
        else:
            return pos[pro_nodes:]
    
    def modelling_loop(self, data, scoring=False, recycle_num=3, dist_threhold=5):
        '''
        generating loop conformations and  predicting the confidence score
        '''
        device = data['protein'].node_s.device
        batch_size = data['protein'].batch.max()+1
        # encoder
        pro_node_s, loop_node_s = self.encoding(data)
        # loop_modelling
        loop_pos = self.loop_modelling(pro_node_s, loop_node_s, data, recycle_num, train=False)  

        # scoring
        if scoring:
            mdn_score = self.scoring(loop_s=loop_node_s, loop_pos=loop_pos, pro_s=pro_node_s, data=data,
                                                               dist_threhold=dist_threhold, batch_size=batch_size, train=False)
        else:
            mdn_score = torch.zeros((batch_size, 1), device=device, dtype=torch.float)
        return loop_pos, mdn_score
    