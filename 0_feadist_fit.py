import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import scipy.io as scio
import sys
from sklearn import preprocessing
from sklearn.decomposition import SparsePCA
import joblib
from feamap.preparation import drug_data, drug_fea, protein_data, protein_fea, fea_statistic, to_dist_matrix



prj_path = Path(__file__).parent.resolve()

def MinMaxScaleClip(x, xmin, xmax):
    # print("MinMaxScaleClip")
    scaled = (x - xmin) / ((xmax - xmin) + 1e-8)
    return scaled.clip(0, 1)

def StandardScaler(x, xmean, xstd):
    # print("StandardScaler")
    return (x-xmean) / (xstd + 1e-8)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached scales")
    parser.add_argument("--scale_method", type=str, default='standard', choices=['standard', 'minmax', 'None'], help="")
    parser.add_argument("--drug_featype", type=str, default='xmol', choices=['xmol'], help="")
    parser.add_argument("--prot_featype", type=str, default='esm2', choices=['esm2'], help="")
    parser.add_argument("--scale_source", type=str, default='bindingdb', help="")
    params = parser.parse_args()
    print(vars(params))

    save_path_df = prj_path / 'data' / 'processed_data' / 'drug_fea' / 'scale' / f'{params.scale_method}'
    save_path_df.mkdir(parents=True, exist_ok=True)
    save_path_pf = prj_path / 'data' / 'processed_data' / 'protein_fea' / 'scale' / f'{params.scale_method}'
    save_path_pf.mkdir(parents=True, exist_ok=True)
    # load data
    print('...loading data...')
    drugs = pd.read_csv(prj_path / 'data' / 'original_data' / 'scale' / f'{params.scale_source}_all-data-merge-drug.csv', index_col=0, low_memory=False)
    # drugs.index = drugid
    # drugs.columns = xmol_H_ids
    num_drugs = len(drugs.index)

    proteins = pd.read_csv(prj_path / 'data' / 'original_data' / 'scale' / f'{params.scale_source}_all-data-merge-prot.csv', index_col=0, low_memory=False)
    # proteins.index = protid
    # proteins.columns = esm2_H_ids
    num_proteins = len(proteins.index)


    # featuregeneration
    # drug feature
    print('...working on drug feature...')
    drugs_fea, bitsinfo_drug = drug_fea(drugs, type=params.drug_featype)
    # drugs_fea.to_csv(save_path_df / f'drug_llmfea_{params.drug_featype}.csv')
    # bitsinfo_drug.to_csv(save_path_df / f'drug_fea_bitsinfo_{params.drug_featype}.csv')

    # protein feature
    print('...working on protein feature...')
    proteins_fea, bitsinfo_prot = protein_fea(proteins, type=params.prot_featype)
    # proteins_fea.to_csv(save_path_pf / f'protein_llmfea_{params.prot_featype}.csv')
    # bitsinfo_prot.to_csv(save_path_pf / f'prot_fea_bitsinfo_{params.prot_featype}.csv')

    # statistic feature
    drugfea_scale = fea_statistic(drugs_fea)
    protfea_scale = fea_statistic(proteins_fea)
    # save data
    drugfea_scale.to_pickle(save_path_df/f'{params.drug_featype}_{params.scale_source}_scale.cfg')
    protfea_scale.to_pickle(save_path_pf/f'{params.prot_featype}_{params.scale_source}_scale.cfg')


    drugs_to_dist, proteins_to_dist=[], []
    if params.scale_method == 'standard':
        print("StandardScaler")
        for id, d_fea in drugs_fea.iterrows():
            drug_to_dist = StandardScaler(d_fea, drugfea_scale['mean'], drugfea_scale['std'])
            drugs_to_dist.append(np.nan_to_num(drug_to_dist))
        drugs_to_dist = np.stack(drugs_to_dist, axis=0)
        for id, c_fea in proteins_fea.iterrows():
            protein_to_dist = StandardScaler(c_fea, protfea_scale['mean'], protfea_scale['std'])
            proteins_to_dist.append(np.nan_to_num(protein_to_dist))
        proteins_to_dist = np.stack(proteins_to_dist, axis=0)

    elif params.scale_method == 'minmax':
        print("MinMaxScaleClip")
        for id, d_fea in drugs_fea.iterrows():
            drug_to_dist = MinMaxScaleClip(d_fea, drugfea_scale['min'], drugfea_scale['max'])
            drugs_to_dist.append(np.nan_to_num(drug_to_dist))
        drugs_to_dist = np.stack(drugs_to_dist, axis=0)
        for id, c_fea in proteins_fea.iterrows():
            protein_to_dist = MinMaxScaleClip(c_fea, protfea_scale['min'], protfea_scale['max'])
            proteins_to_dist.append(np.nan_to_num(protein_to_dist))
        proteins_to_dist = np.stack(proteins_to_dist, axis=0)
        
    elif params.scale_method == 'None':
        drugs_to_dist = drugs_fea.values
        proteins_to_dist = proteins_fea.values

    # feature bits dist define
    pd.DataFrame(drugs_to_dist,columns=drugs_fea.columns,index=drugs_fea.index).to_csv(save_path_df / f'drug_normalizedfea_{params.drug_featype}_{params.scale_source}.csv')
    pd.DataFrame(proteins_to_dist,columns=proteins_fea.columns,index=proteins_fea.index).to_csv(save_path_pf / f'protein_normalizedfea_{params.prot_featype}_{params.scale_source}.csv')
    print('to_dist_matrix for drug')
    to_dist_matrix(drugs_to_dist, 'drug_fea', bitsinfo_drug['IDs'], f'{params.scale_method}/{params.drug_featype}_{params.scale_source}', methods = ['cosine', 'correlation', 'jaccard'])
    print('to_dist_matrix for protein')
    to_dist_matrix(proteins_to_dist, 'protein_fea', bitsinfo_prot['IDs'], f'{params.scale_method}/{params.prot_featype}_{params.scale_source}', methods = ['cosine', 'correlation', 'jaccard'])

    # if params.overwrite_cache:
    #     import os
    #     from shutil import copy
    #     for file in os.listdir(save_path_df):
    #         if os.path.splitext(str(file))[1]=='.cfg':
    #             copy(os.path.join(save_path_df,file), os.path.join(prj_path,'feamap','config',f'trans_from_{str(file).split('_')[1]}'))
    #     for file in os.listdir(save_path_pf):
    #         if os.path.splitext(str(file))[1]=='.cfg':
    #             copy(os.path.join(save_path_pf,file), os.path.join(prj_path,'feamap','config',f'trans_from_{str(file).split('_')[1]}'))
    








