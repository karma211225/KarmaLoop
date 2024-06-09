#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @author : Xujun Zhang, Tianyue Wang

'''
@File    :   loop_generating.py
@Time    :   2023/03/09 19:31:32
@Author  :   Xujun Zhang, Tianyue Wang
@Version :   1.0
@Contact :   
@License :   
@Desc    :   None
'''
# here put the import lib
import argparse
import os
import sys
import time
import numpy as np
import torch.nn as nn
import pandas as pd
import torch.optim
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
# dir of current
pwd_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.dirname(pwd_dir)
sys.path.append(project_dir)
from utils.fns import Early_stopper, set_random_seed, save_loop_file
from dataset.graph_obj import LoopGraphDataset
from dataset.dataloader_obj import PassNoneDataLoader
from architecture.Net_architecture import KarmaLoop

class DataLoaderX(PassNoneDataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

# get parameters from command line
argparser = argparse.ArgumentParser()
argparser.add_argument('--graph_file_dir', type=str,
                        default='/root/surface_graph',
                       help='the graph files path')
argparser.add_argument('--out_dir', type=str,
                       default='/root/CASP_result',
                       help='dir for recording loop conformations and scores')
argparser.add_argument('--multi_conformation', action='store_true',
                       help='whether predict miltiple loop conformations')
argparser.add_argument('--scoring', action='store_true',
                       help='whether predict loop generating scores')
argparser.add_argument('--save_file', action='store_true',
                       help='whether save predicted loop conformations')
argparser.add_argument('--batch_size', type=int,
                       default=128,
                       help='batch size')
argparser.add_argument('--random_seed', type=int,
                       default=2020,
                       help='random_seed')
args = argparser.parse_args()


def get_multi_pos_sr(rmsds_r, threshold=2.0):
    sr_multipose = (np.stack(rmsds_r, axis=0) <= threshold).T.any(axis=1)
    sr_multipose = sr_multipose.sum() / sr_multipose.shape[0]
    return sr_multipose

# set random seed
set_random_seed(args.random_seed)
# get pdb_ids
pdb_ids = [graphs.split('.')[0] for graphs in os.listdir(args.graph_file_dir)]
# dataset
test_dataset = LoopGraphDataset(graph_dir=args.graph_file_dir, 
                                pdb_ids=pdb_ids, 
                                geometric_pos_init='anchor', 
                                multi_conformation=args.multi_conformation, 
                                sigma=1.75)
# dataloader
test_dataloader = DataLoaderX(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, follow_batch=['atom2nonloopres'], pin_memory=True)
# device
device_id = 0
if torch.cuda.is_available():
    my_device = f'cuda:{device_id}'
else:
    my_device = 'cpu'
# model
model = KarmaLoop(hierarchical=True)
model = nn.DataParallel(model, device_ids=[device_id], output_device=device_id)
model.to(my_device)
# stoper
print('########### Start modelling loop ###########')
print(f'pdb_ids: {len(test_dataset)}')
print('# load model')
# load existing model
model_file = f'{project_dir}/model_pkls/karmaloop.pkl'
# 
model.load_state_dict(torch.load(model_file, map_location=my_device)['model_state_dict'], strict=True)
# time
start_time = time.perf_counter()
total_time = 0
data_statistic = []
time_info=[]
rmsds_r = []
repeats = 5 if args.multi_conformation else 3
for re in range(repeats):
    set_random_seed(re)
    rmsds = torch.as_tensor([]).to(my_device)
    confidence_scores = []
    pdb_ids = []
    ff_corrected_rmsds = []
    align_corrected_rmsds = []
    model.eval()
    egnn_time = 0
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_dataloader)):
            # to device
            data = data.to(my_device)
            start_ = time.time()
            # forward
            egnn_start_time = time.perf_counter()
            pos_pred, mdn_score = model.module.modelling_loop(data, scoring=args.scoring, recycle_num=3, dist_threhold=5)
            egnn_time += time.perf_counter() - egnn_start_time
            pos_true = data['loop'].xyz
            batch = data['loop'].batch
            end_=time.time()
            per_time = end_-start_
            time_info.append([data.pdb_id,per_time])
            pos_loss = model.module.cal_rmsd(pos_true, pos_pred, batch) 
            rmsds = torch.cat([rmsds, pos_loss], dim=0)
            # output conformation
            if args.save_file:
                out_dir_r = f'{args.out_dir}/{re}'
                os.makedirs(out_dir_r, exist_ok=True)
                data.pos_preds = pos_pred
                save_loop_file(data, out_dir=out_dir_r, out_init=False)
            pdb_ids.extend(data.pdb_id)
            confidence_scores.extend(mdn_score.cpu().numpy().tolist())
        if args.scoring:
            df = pd.DataFrame({'pdb_id':pdb_ids, 'confidence score': confidence_scores})
            df.to_csv(f'{args.out_dir}/{re}_score.csv', index=False)
        # report 
        data_statistic.append([rmsds.mean(), 
                               rmsds.median(), 
                               rmsds.max(), 
                               rmsds.min(), 
                               (rmsds<=5).sum()/rmsds.size(0), 
                               (rmsds<=4.5).sum()/rmsds.size(0), 
                               (rmsds<=4).sum()/rmsds.size(0), 
                               (rmsds<=3.5).sum()/rmsds.size(0), 
                               (rmsds<=3).sum()/rmsds.size(0), 
                               (rmsds<=2.5).sum()/rmsds.size(0), 
                               (rmsds<=2).sum()/rmsds.size(0), 
                               (rmsds<=1).sum()/rmsds.size(0), 
                               egnn_time / 60, 
                               ])
        rmsds_r.append(rmsds.cpu().numpy())
data_statistic_mean = torch.as_tensor(data_statistic).mean(dim=0)
data_statistic_std = torch.as_tensor(data_statistic).std(dim=0)

prediction_time = time.perf_counter()
print(f'''Total Time: {(prediction_time - start_time) / 60} min
Sample Num: {len(test_dataset)}
Multi Conformation Num: {repeats}
# prediction
Time Spend: {data_statistic_mean[12]} ± {data_statistic_std[12]} min
Mean RMSD: {data_statistic_mean[0]} ± {data_statistic_std[0]}   
Medium RMSD: {data_statistic_mean[1]} ± {data_statistic_std[1]}
Max RMSD: {data_statistic_mean[2]} ± {data_statistic_std[2]}
Min RMSD: {data_statistic_mean[3]} ± {data_statistic_std[3]}
Success RATE(5A): {data_statistic_mean[4]} ± {data_statistic_std[4]})
Success RATE(4.5A): {data_statistic_mean[5]} ± {data_statistic_std[5]})
Success RATE(4A): {data_statistic_mean[6]} ± {data_statistic_std[6]})
Success RATE(3.5A): {data_statistic_mean[7]} ± {data_statistic_std[7]})
Success RATE(3A): {data_statistic_mean[8]} ± {data_statistic_std[8]})
Success RATE(2.5A): {data_statistic_mean[9]} ± {data_statistic_std[9]})
Success RATE(2A): {data_statistic_mean[10]} ± {data_statistic_std[10]})
Success RATE(1A): {data_statistic_mean[11]} ± {data_statistic_std[11]}
# multi pose
Success RATE(5A): {get_multi_pos_sr(rmsds_r, threshold=5)} 
Success RATE(4.5A): {get_multi_pos_sr(rmsds_r, threshold=4.5)}
Success RATE(4A): {get_multi_pos_sr(rmsds_r, threshold=4)}
Success RATE(3.5A): {get_multi_pos_sr(rmsds_r, threshold=3.5)}
Success RATE(3A): {get_multi_pos_sr(rmsds_r, threshold=3)}
Success RATE(2.5A): {get_multi_pos_sr(rmsds_r, threshold=2.5)}
Success RATE(2A): {get_multi_pos_sr(rmsds_r, threshold=2)}
Success RATE(1A):  {get_multi_pos_sr(rmsds_r, threshold=1)} 
''')

