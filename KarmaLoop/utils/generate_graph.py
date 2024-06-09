#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @author : Xujun Zhang, Tianyue Wang

'''
@File    :   generate_graph.py
@Time    :   2023/02/24 14:21:01
@Author  :   Xujun Zhang, Tianyue Wang
@Version :   1.0
@Contact :   
@License :   
@Desc    :   None
'''

# here put the import lib
import os
import sys
import glob
import pandas as pd
import argparse
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from dataset import graph_obj

argparser = argparse.ArgumentParser()
argparser.add_argument('--protein_file_dir', type=str, default='/root/KarmaLoop/example/single_example/raw')
argparser.add_argument('--pocket_file_dir', type=str, default='/root/surface_pocket')
argparser.add_argument('--graph_file_dir', type=str, default='/root/surface_graph')
args = argparser.parse_args()
# init
pocket_path = args.pocket_file_dir
pdb_ids = ['_'.join(pdb.split('_')[:4]) for pdb in os.listdir(pocket_path)]
graph_file_dir = args.graph_file_dir
os.makedirs(graph_file_dir, exist_ok=True)
print('########### Start generating graph ###########')
print(f'pdb_ids: {len(pdb_ids)}')
# 
test_dataset = graph_obj.LoopGraphDataset(graph_dir=graph_file_dir, 
                                          pdb_ids=pdb_ids)
test_dataset.generate_graph(pocket_path, args.protein_file_dir, n_job=-1, verbose=True)
