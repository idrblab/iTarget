import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import tqdm
import scipy.io as scio
import sys
from sklearn import preprocessing
from sklearn.decomposition import SparsePCA
import joblib

from feamap import basemap
from feamap import load_config
from feamap.preparation import drug_data, drug_fea, protein_data, protein_fea

prj_path = Path(__file__).parent.resolve()
save_path = prj_path / 'data' / 'processed_data'
save_path.mkdir(parents=True, exist_ok=True)
save_path_df = prj_path / 'data' / 'processed_data' / 'drug_fea' / 'map_transferred'
save_path_df.mkdir(parents=True, exist_ok=True)
save_path_pf = prj_path / 'data' / 'processed_data' / 'protein_fea' / 'map_transferred'
save_path_pf.mkdir(parents=True, exist_ok=True)

def fit_map(disttype = 'ALL', datatype = 'protein_fea', ftype = 'esm2', flist=None, split_channels=True, metric = 'cosine', fitmethod = 'umap'):
    # mp = basemap.Map(disttype = disttype, datatype = datatype, ftype = ftype, flist = flist, fmap_type='grid', split_channels=split_channels,  metric=metric, var_thr=0)
    mp = basemap.Map(disttype = disttype, datatype = datatype, ftype = ftype, flist = flist, fmap_type='grid', split_channels=split_channels,  metric=metric, var_thr=1e-4)
    mp.fit(method = fitmethod, min_dist = 0.1, n_neighbors = 30, random_state=1)
    # Visulization and save your fitted map
    # mp.plot_scatter(htmlpath=save_path, htmlname=None)
    mp.plot_grid(htmlpath=save_path, htmlname=None)
    mp.save(save_path / f'{ftype}_fitted.mp')
    return mp

def load_map(disttype = 'ALL', datatype = 'protein_fea', ftype = 'esm2', flist=None, split_channels=True, metric = 'cosine', fitmethod = 'umap'):
    # mp = basemap.Map(disttype = disttype, datatype = datatype, ftype = ftype, flist = flist, fmap_type='grid', split_channels=split_channels,  metric=metric, var_thr=0)
    mp = basemap.Map(disttype = disttype, datatype = datatype, ftype = ftype, flist = flist, fmap_type='grid', split_channels=split_channels,  metric=metric, var_thr=1e-4)
    mp = mp.load(save_path / f'{ftype}_fitted.mp')
    # Visulization and save your fitted map
    # mp.plot_scatter(htmlpath=save_path, htmlname=None)
    mp.plot_grid(htmlpath=save_path, htmlname=None)
    return mp

def trans_map(params, data_df, mp, datatype, ftype):
    if datatype == 'drug_fea':
        fea_o, bitsinfo_drug = drug_fea(data_df, type=ftype)
        fea_o.to_csv(save_path_df / f'drug_llmfea_{ftype}_beforetrans.csv')
        bitsinfo_drug.to_csv(save_path_df / f'drug_fea_bitsinfo_{ftype}_beforetrans.csv')
    elif datatype == 'protein_fea':
        fea_o, bitsinfo_prot = protein_fea(data_df, type=ftype)
        fea_o.to_csv(save_path_pf / f'protein_llmfea_{ftype}_beforetrans.csv')
        bitsinfo_prot.to_csv(save_path_pf / f'prot_fea_bitsinfo_{ftype}_beforetrans.csv')
        
    fea_map, ids = mp.batch_transform(arrs=fea_o, scale=True, scale_method=params.scale_method)
        # fea_map = np.stack((fea_map, fea))
    print('fea_map.shape', fea_map.shape)
    return fea_map.astype('float32'), ids






if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--ftype", type=str, default='esm2', choices=['esm2', 'xmol', ], help="")
    parser.add_argument("--datatype", type=str, default='protein_fea', choices=['drug_fea', 'protein_fea'], help="")
    parser.add_argument("--disttype", type=str, default='ALL', choices=['ALL','chembl','bindingdb','biosnap','celegans','human','uniprot+fullchembl'], help="the data source that the task working on")
    parser.add_argument("--fea_list", type=list, default=[],help="")
    parser.add_argument("--metric", type=str, default='cosine', choices=['correlation', 'cosine', 'jaccard'], help="")
    parser.add_argument("--fitmethod", type=str, default='umap', choices=['umap', 'tsne', 'mds'], help="")
    parser.add_argument("--channel", type=str, default='False', choices=['True', 'False'], help="")
    parser.add_argument("--scale_method", type=str, default='standard', choices=['standard', 'minmax', 'None'], help="")
    parser.add_argument("--source", type=str, default='bindingdb', help="the data source that the task working on")
    params = parser.parse_args()
    print(vars(params))

    if params.channel=='True':
        split_channels = True
    elif params.channel=='False':
        split_channels = False

    if params.datatype == 'drug_fea':
        # load map
        mp_d = fit_map(disttype = params.disttype, datatype = 'drug_fea', ftype = params.ftype, flist=[], split_channels=split_channels, metric = params.metric, fitmethod = params.fitmethod)
        # transform self defined data
        # load data to trans
        drugs, num_drugs = drug_data(params.source)
        drugmap, ids = trans_map(params, drugs, mp_d, datatype = 'drug_fea', ftype = params.ftype)
        # save data
        with open(save_path_df/'drug_fea.npy', 'wb') as f:
            np.save(f, drugmap)
        pd.DataFrame(ids).to_csv(save_path_df/'drug_list.csv')
        # pd.DataFrame(drugmap).to_csv(save_path_pf/'drug_fea.csv')

    elif params.datatype == 'protein_fea':
        # load map
        mp_p = fit_map(disttype = params.disttype, datatype = 'protein_fea', ftype = params.ftype, flist=[], split_channels=split_channels, metric = params.metric, fitmethod = params.fitmethod)
        # transform self defined data
        # load data to trans
        proteins, num_proteins = protein_data(params.source)
        protmap, ids = trans_map(params, proteins, mp_p, datatype = 'protein_fea', ftype = params.ftype)
        # save data
        with open(save_path_pf/'protein_fea.npy', 'wb') as f:
            np.save(f, protmap)
        pd.DataFrame(ids).to_csv(save_path_pf/'prot_list.csv')
        # pd.DataFrame(protmap).to_csv(save_path_pf/'protein_fea.csv')
    
