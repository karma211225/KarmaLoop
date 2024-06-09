#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2023/10
# @author : Xujun Zhang, Tianyue Wang
import argparse
import os
import sys
from multiprocessing import Pool
import numpy as np
from prody import parsePDB, writePDB
import pandas as pd
pwd_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(pwd_dir)


def get_pure_protein(pdb_id):
    '''
    remove water, ligand, ion, etc. from protein
    retain only heavy atoms in protein
    '''
    raw_protein = f'{dataset_path}/{pdb_id}_init.pdb'
    dst_protein = f'{dst_path}/{pdb_id}.pdb'
    protein_prody_obj = parsePDB(raw_protein)
    selected = protein_prody_obj.select('protein and not hydrogen')
    writePDB(dst_protein, atoms=selected)

def get_loop_pocket(pdb_id):
    '''
    get lprotein pocket based on reference loop
    and move random generate loop pose to the center of pocket
    :param pdb_id:
    :return:
    '''
    pdbid, chain, res_num_src, res_num_dst = pdb_id.split('_')
    # src
    # protein_pdb = f'{dataset_path}/{pdbid}.pdb'
    protein_pdb = f'{dataset_path}/{pdb_id}.pdb'
    # dst
    pocket_pdb = f'{dst_path}/{pdbid}_{chain}_{res_num_src}_{res_num_dst}_pocket_12A.pdb'
    # select & save pocket
    if not os.path.exists(pocket_pdb) or os.path.getsize(pocket_pdb) == 0:
        protein_prody_obj = parsePDB(protein_pdb)
        chain_obj = protein_prody_obj.select(f'chain {chain}')
        # in case the chain name is lacked
        if chain_obj is not None:
            ligpos = chain_obj.select(f'resnum {int(res_num_src)}:{int(res_num_dst)}').getCoords()
        else:
            ligpos = protein_prody_obj.select(f'resnum {int(res_num_src)}:{int(res_num_dst)}').getCoords()
        condition = 'same residue as exwithin 15 of somepoint'
        pocket_selected = protein_prody_obj.select(condition,
                                                   somepoint=ligpos)  # （n, 3）
        writePDB(pocket_pdb, atoms=pocket_selected)


def is_inside_matrix(test_points, center, boundary_points):
    """
    Determines if a set of test points are inside a matrix defined by a center point and boundary points.

    Args:
        test_points (numpy.ndarray): An array of test points to check.
        center (numpy.ndarray): The center point of the matrix.
        boundary_points (numpy.ndarray): An array of boundary points that define the matrix.

    Returns:
        numpy.ndarray: A boolean array indicating whether each test point is inside the matrix.
    """
    test_points = np.array(test_points)
    center = np.array(center)
    boundary_points = np.array(boundary_points)

    vectors_test = test_points - center
    vectors_boundary = boundary_points - center

    dot_products = np.dot(vectors_test, vectors_boundary.T)
    return (dot_products > 0).sum(axis=1) >= max(3, boundary_points.shape[0]//3)

def check_points_matrix_mask(points_to_test, center, boundary_points):
    """
    Check if the given points are inside the boundary points matrix.

    Args:
        points_to_test (numpy.ndarray): The points to test.
        center (tuple): The center of the boundary points matrix.
        boundary_points (numpy.ndarray): The boundary points matrix.

    Returns:
        numpy.ndarray: A boolean mask indicating which points are inside the boundary points matrix.
    """
    mask_inside = is_inside_matrix(points_to_test, center, boundary_points)
    return mask_inside


def check_overlap(new_selection, old_selection):
    new_key = set(['_'.join(i) for i in np.stack([new_selection.getChids(),
                                                new_selection.getResnums(),
                                                new_selection.getIcodes()], axis=0).T])
    old_key = set(['_'.join(i) for i in np.stack([old_selection.getChids(),
                                                old_selection.getResnums(),
                                                old_selection.getIcodes()], axis=0).T])
    # get the overlap
    overlap = new_key & old_key
    # get the overlap rate
    overlap_rate = len(overlap) / len(old_key)
    return len(new_key), len(overlap), len(old_key), overlap_rate


def get_loop_surface_pocket(pdb_id):
    try:
        pdbid, chain, res_num_src, res_num_dst = pdb_id.split('_')
        res_num_src, res_num_dst = int(res_num_src), int(res_num_dst)
        loop_len = int(res_num_dst) - int(res_num_src) + 1
        # src
        protein_pdb = f'{dataset_path}/{pdbid}.pdb'
        # dst
        pocket_pdb = f'{dst_path}/{pdbid}_{chain}_{res_num_src}_{res_num_dst}_pocket_12A.pdb'
        # select & save pocket
        protein_prody_obj = parsePDB(protein_pdb)
        loop_mol = protein_prody_obj.select(f'chain {chain} and (resnum {res_num_src} to {res_num_dst})')
        loop_res_num = len(set(list(zip(loop_mol.getResnums(), loop_mol.getIcodes()))))
        if loop_res_num != loop_len:
            print(f"Warning: For protein {pdb_id}, the number of residues in the loop does not match the count derived from the loop's start and end indices. This discrepancy may be acceptable if the protein functions as an antigen or antibody. Otherwise, a thorough review of this protein is advised.")
            loop_len = loop_res_num
        # get nearby residue
        nearby_resid_src = res_num_src - 1
        nearby_resid_dst = res_num_dst + 1
        nearby_res_src_obj = protein_prody_obj.select(f'chain {chain} and resnum {int(nearby_resid_src)}')
        nearby_res_dst_obj = protein_prody_obj.select(f'chain {chain} and resnum {int(nearby_resid_dst)}')
        nearby_res_src = nearby_res_src_obj.getCoords()
        nearby_res_dst = nearby_res_dst_obj.getCoords()
        # 计算中点
        mid_point = (nearby_res_src.mean(axis=0) + nearby_res_dst.mean(axis=0))/2
        # radius
        nearby_res_radius = np.linalg.norm(nearby_res_src.mean(axis=0) - nearby_res_dst.mean(axis=0)) / 2 + 1
        loop_radius = min(2.00 * loop_len, 16.0) 
        radius = max(nearby_res_radius, loop_radius)
        if radius > 20:
            delta_x = nearby_res_dst[:3] - nearby_res_src[:3]
            psuedo_pos = [nearby_res_src[:3] + t*delta_x for t in np.linspace(0, 1, (loop_len)+2)]
            psuedo_pos = np.concatenate(psuedo_pos, axis=0)
            pocket_selected = protein_prody_obj.select(f'{loop_mol.getSelstr()} or (same residue as exwithin 15 of somepoint)',
                                                    somepoint=psuedo_pos)
        else:
            pocket_selected = protein_prody_obj.select( f'same residue as exwithin {radius} of somepoint', somepoint=mid_point)  # （n, 3）
            bigger_pocket_selected = protein_prody_obj.select( f'same residue as exwithin {radius+12} of somepoint', somepoint=mid_point)
            # # compare the atom difference between pocket_selected and bigger_pocket_selected
            # get the difference
            diff_pocket_selected = protein_prody_obj.select(f'not ({pocket_selected.getSelstr()}) and ({bigger_pocket_selected.getSelstr()})')
            diff_pocket_selected_keys = ['_'.join(i) for i in np.stack([diff_pocket_selected.getChids(), 
                                                    diff_pocket_selected.getResnums()], axis=0).T]
            # get mask
            mask = check_points_matrix_mask(diff_pocket_selected.getCoords(), mid_point, pocket_selected.getCoords())
            outside_residues = np.asarray(diff_pocket_selected_keys)[~mask]
            outside_residues = list(set(outside_residues))
            if len(outside_residues) > 0:
                outside_residues = [i.split('_') for i in outside_residues]
                add_outside_residues_string = 'or'.join([f' (chain {i[0]} and resnum {i[1]}) ' for i in outside_residues])
                pocket_selected = protein_prody_obj.select(f'{pocket_selected.getSelstr()} or {add_outside_residues_string}')
            # get smaller pocket
            pocket_selected = protein_prody_obj.select(f'{pocket_selected.getSelstr()} or (same residue as exwithin {min(max(abs(16-radius), 5), 7)} of somepoint) or {loop_mol.getSelstr()} or {nearby_res_src_obj.getSelstr()} or {nearby_res_dst_obj.getSelstr()}', somepoint=pocket_selected.getCoords())
            # pocket_selected = pocket_selected + pocket_supplement
        # output
        writePDB(pocket_pdb, atoms=pocket_selected)
    except Exception as e:
        print(f'{pdb_id} error due to {e}')


def pipeline(pdb_id):
    try:
        get_loop_surface_pocket(pdb_id)
    except Exception as e:
        print(f'{pdb_id} error due to {e}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--protein_file_dir', type=str, default='/root/KarmaLoop/example/single_example/raw')
    argparser.add_argument('--pocket_file_dir', type=str, default='/root/KarmaLoop/example/single_example/pocket')
    argparser.add_argument('--csv_file', type=str, default='/root/KarmaLoop/example/single_example/example.csv')

    args = argparser.parse_args()
    dataset_path = args.protein_file_dir
    dst_path = args.pocket_file_dir
    csv_file = args.csv_file
    # mkdir
    os.makedirs(dst_path, exist_ok=True)
    # read pdb_id
    df = pd.read_csv(csv_file)
    pdb_ids = df.loc[:, 'name'].values
    print('########### Start pocket extraction ###########')
    print(f'pdb_ids: {len(pdb_ids)}')
    # multiprocessing
    pool = Pool()
    pool.map(pipeline, pdb_ids)
    pool.close()
    pool.join()
