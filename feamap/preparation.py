from bisect import bisect
from typing import DefaultDict
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import column
import tqdm
import scipy.io as scio
import sys
import sklearn
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from joblib import Parallel, delayed
from collections import defaultdict
from collections import OrderedDict
import seaborn as sns
import math


from feamap.utils import distances, calculator
from feamap import summary

prj_path = Path(__file__).parent.resolve().parent.resolve()

def pair_statistic(label, type):
    drugs = label['drugid'].value_counts(sort=False)
    drugs.name = 'pair_counts'
    drugs = drugs.to_frame().assign(pdb_counts = np.nan, uniprot_counts = np.nan, pos_counts = np.nan, neg_counts = np.nan)
    for drug in drugs.index:
        x = label.loc[label['drugid']==drug]
        source_counts = x['source'].value_counts(sort=False).to_dict()
        posneg_counts = x['label'].value_counts(sort=False).to_dict()
        drugs.at[drug,'pdb_counts'] = source_counts.get('pdb')
        drugs.at[drug,'uniprot_counts'] = source_counts.get('uniprot')
        drugs.at[drug,'pos_counts'] = posneg_counts.get(1)
        drugs.at[drug,'neg_counts'] = posneg_counts.get(0)
    drugs.to_csv(prj_path / 'data' / f'original_data' / f'drugpair_statistic_{type}.csv')

    proteins = label['protid'].value_counts(sort=False)
    proteins.name = 'pair_counts'
    proteins = proteins.to_frame().assign(pdb_counts = np.nan, uniprot_counts = np.nan, pos_counts = np.nan, neg_counts = np.nan)
    for protein in proteins.index:
        x = label.loc[label['protid']==protein]
        source_counts = x['source'].value_counts(sort=False).to_dict()
        posneg_counts = x['label'].value_counts(sort=False).to_dict()
        proteins.at[protein,'pdb_counts'] = source_counts.get('pdb')
        proteins.at[protein,'uniprot_counts'] = source_counts.get('uniprot')
        proteins.at[protein,'pos_counts'] = posneg_counts.get(1)
        proteins.at[protein,'neg_counts'] = posneg_counts.get(0)
    proteins.to_csv(prj_path / 'data' / f'original_data' / f'proteinpair_statistic_{type}.csv')

def pair_DOWNsampling(params, label):
    _label = []
    for drug in label['drugid'].drop_duplicates():
        x = label.loc[label['drugid']==drug]
        pos = x.loc[x['label']==1]
        neg = x.loc[x['label']==0]
        if pos.shape[0] < neg.shape[0]:
            neg = neg.sample(n=pos.shape[0],random_state=params.random_seed)
        elif pos.shape[0] > neg.shape[0]:
            pos = pos.sample(n=neg.shape[0],random_state=params.random_seed)
        _label.append(shuffle(pd.concat([pos, neg]), random_state = params.random_seed))
    _label = shuffle(pd.concat(_label), random_state = params.random_seed)
    print(f"sample num after DOWN-sampling: {_label.shape[0]}")

    return _label

def data_washing(params, label):
    drugpair_statistic = pd.read_csv(prj_path / 'data' / 'original_data' / 'drugpair_statistic_original.csv', index_col=0)
    proteinpair_statistic = pd.read_csv(prj_path / 'data' / 'original_data' / 'proteinpair_statistic_original.csv', index_col=0)

    if params.data_wash:
        print("WASHING DATA")
        print(f"WASHING datasets based on drugs")
        print(f"sample num before washing: {label.shape[0]}")

        if params.sampling=='DOWN':
            print(f"len(label) before Down sampling: {len(label)}")
            label = pair_DOWNsampling(params, label)
            print(f"len(label) after Down sampling: {len(label)}")

        print(f"DROP drugs with pos/neg samples'count < 5, these no-trained drugs' samples would be regarded as individual tests")
        print(f"DROP proteins with pos/neg samples'count < 5, these no-trained proteins' samples would be regarded as individual tests")
        print(f"num_drugs before dropping: {drugpair_statistic.shape[0]}")
        print(f"num_proteins before dropping: {proteinpair_statistic.shape[0]}")
        drugs_for_test = drugpair_statistic.loc[(drugpair_statistic['pos_counts']<5) | (drugpair_statistic['neg_counts']<5)]
        proteins_for_test = proteinpair_statistic.loc[(proteinpair_statistic['pos_counts']<5) | (proteinpair_statistic['neg_counts']<5)]
        # drugs_for_test = drugpair_statistic.loc[(drugpair_statistic['pos_counts']+drugpair_statistic['neg_counts'])<=100]
        # proteins_for_test = proteinpair_statistic.loc[(proteinpair_statistic['pos_counts']+proteinpair_statistic['neg_counts'])<=100]
        print(f"num_drugs after dropping: {drugpair_statistic.shape[0]-drugs_for_test.shape[0]}")
        print(f"num_proteins after dropping: {proteinpair_statistic.shape[0]-proteins_for_test.shape[0]}")

        print(f"len(label): {len(label)}")
        # for_test = pd.concat([drugs_for_test,])
        # for_test = pd.concat([proteins_for_test,])
        for_test = pd.concat([drugs_for_test,proteins_for_test])
        # x_test = label.loc[(label['drugid'].isin(for_test.index))|(label['protid'].isin(for_test.index))]
        # x_trainval = label.loc[~((label['drugid'].isin(for_test.index))|(label['protid'].isin(for_test.index)))]
        _x_test = label.loc[(label['drugid'].isin(for_test.index))|(label['protid'].isin(for_test.index))]
        label = label.loc[~((label['drugid'].isin(for_test.index))|(label['protid'].isin(for_test.index)))]
        # print(f"len(x_test),len(x_trainval): {len(x_test),len(x_trainval)}")
        print(f"len(_x_test),len(label): {len(_x_test),len(label)}")

    # # reversal TEST set
    # # drugs_for_postest = drugpair_statistic.sample(int(0.1*len(drugpair_statistic)), random_state=params.random_seed)
    # # drugs_for_negtest = drugpair_statistic.loc[~(drugpair_statistic.index.isin(drugs_for_postest.index))].sample(int(0.1*len(drugpair_statistic)), random_state=params.random_seed)
    # # x_test = label.loc[((label['drugid'].isin(drugs_for_postest.index))&(label['label']==1))|((label['drugid'].isin(drugs_for_negtest.index))&(label['label']==0))]
    # # x_trainval = label.loc[~(((label['drugid'].isin(drugs_for_postest.index))&(label['label']==1))|((label['drugid'].isin(drugs_for_negtest.index))&(label['label']==0)))]
    # proteins_for_postest = proteinpair_statistic.sample(int(0.1*len(proteinpair_statistic)), random_state=params.random_seed)
    # proteins_for_negtest = proteinpair_statistic.loc[~(proteinpair_statistic.index.isin(proteins_for_postest.index))].sample(int(0.1*len(proteinpair_statistic)), random_state=params.random_seed)
    # x_test = label.loc[((label['protid'].isin(proteins_for_postest.index))&(label['label']==1))|((label['protid'].isin(proteins_for_negtest.index))&(label['label']==0))]
    # x_trainval = label.loc[~(((label['protid'].isin(proteins_for_postest.index))&(label['label']==1))|((label['protid'].isin(proteins_for_negtest.index))&(label['label']==0)))]
    # print(f"len(label): {len(label)}")
    # print(f"len(x_test),len(x_trainval): {len(x_test),len(x_trainval)}")

    # randomly test setting
    x_trainval, x_test, y_trainval, y_test = train_test_split(label, label['label'], test_size=0.1, random_state=params.random_seed, shuffle=True, stratify=label['label'])
    print(f"len(x_test),len(x_trainval): {len(x_test),len(x_trainval)}")
    savecv_path = prj_path / 'data' / 'processed_data' / 'split_cvdata'
    savecv_path.mkdir(parents=True, exist_ok=True)
    try:_x_test.to_csv(savecv_path / 'test_poneg<5.csv' )
    except: pass
    pair_statistic(x_trainval,'washedtrainval')
    pair_statistic(x_test,'washedtest')

    return x_trainval, x_test


def drug_data(data_source):
    if ',' not in data_source:
        drugs = pd.read_csv(prj_path / 'data' / 'original_data' / f'{data_source}_all-data-merge-drug.csv', index_col=0, low_memory=False)
        # drugs.index = drugid
        # drugs.columns = xmol_H_ids
        drugs.index = f'{data_source}_'+drugs.index
        num_drugs = len(drugs.index)
        return drugs, num_drugs
    elif set(data_source.split(',')) == set(['chembl','ncb']):
        drugss = []
        for ds in data_source.split(','):
            drugs = pd.read_csv(prj_path / 'data' / 'original_data' / f'{ds}_all-data-merge-drug.csv', index_col=0, low_memory=False)
            drugs.index = f'{ds}_'+drugs.index
            drugss.append(drugs)
        drugs = pd.concat(drugss)
        num_drugs = len(drugs.index)
        return drugs, num_drugs
    else: raise ValueError('data_source error')

def protein_data(data_source):
    if ',' not in data_source:
        proteins = pd.read_csv(prj_path / 'data' / 'original_data' / f'{data_source}_all-data-merge-prot.csv', index_col=0, low_memory=False)
        # proteins.index = protid
        # proteins.columns = esm2_H_ids
        proteins.index = f'{data_source}_'+proteins.index
        num_proteins = len(proteins.index)
        return proteins, num_proteins
    elif set(data_source.split(',')) == set(['chembl','ncb']):
        proteinss = []
        for ds in data_source.split(','):
            proteins = pd.read_csv(prj_path / 'data' / 'original_data' / f'{ds}_all-data-merge-prot.csv', index_col=0, low_memory=False)
            proteins.index = f'{ds}_'+proteins.index
            proteinss.append(proteins)
        proteins = pd.concat(proteinss)
        num_proteins = len(proteins.index)
        return proteins, num_proteins
    else: raise ValueError('data_source error')

def relationship_data(data_source):
    if ',' not in data_source:
        label = pd.read_csv(prj_path / 'data' / 'original_data' / f'{data_source}_prot-drug-index-source.csv', low_memory=False)
        pair_statistic(label, 'original')
        return label
    elif set(data_source.split(',')) == set(['chembl','ncb']):
        label = pd.concat([pd.read_csv(prj_path / 'data' / 'original_data' / f'{ds}_prot-drug-index-source.csv', low_memory=False) for ds in data_source.split(',')])
        # pair_statistic(label, 'original')
        return label
    else: raise ValueError('data_source error')

def split_data(params, label, ):
    if ',' not in params.source:
        _x_trainval, _x_test = data_washing(params, label)
        print(f"Split train, valid and test datasets")
        sources = label['source'].drop_duplicates()
        # X = pd.DataFrame([])
        cv_data = defaultdict(list)
        for s in sources:
            x_trainval = _x_trainval.loc[_x_trainval['source']==s]
            y_trainval = x_trainval['label']
            print('np.count_nonzero(y_trainval)/len(y_trainval)', s, ": ", np.count_nonzero(y_trainval)/len(y_trainval))
            cv_data_s = defaultdict(list)
            skf = StratifiedKFold(n_splits=params.kfold_num, shuffle=True, random_state = params.random_seed)
            for k, (t,v) in enumerate(skf.split(x_trainval, y_trainval)):
                cv_data_s[k]=(x_trainval.iloc[t],x_trainval.iloc[v])

            cv_data[s]=cv_data_s

        for k in range(params.kfold_num):
            train_k = shuffle(pd.concat([cv_data[s][k][0] for s in sources]), random_state = params.random_seed)
            valid_k = shuffle(pd.concat([cv_data[s][k][1] for s in sources]), random_state = params.random_seed)
            test_k = shuffle(_x_test, random_state = params.random_seed)
            savecv_path = prj_path / 'data' / 'processed_data' / 'split_cvdata' / f'{params.source}' / f'{k}th_fold'
            savecv_path.mkdir(parents=True, exist_ok=True)
            train_k.to_csv(savecv_path / 'train_k.csv' )
            valid_k.to_csv(savecv_path / 'valid_k.csv' )
            test_k.to_csv(savecv_path / 'test_k.csv' )

    elif set(params.source.split(',')) == set(['chembl','ncb']):
        print(f"Split train, valid and test datasets")
        label['drugid'] = label['source'] + '_' + label['drugid']
        label['protid'] = label['source'] + '_' + label['protid']
        for k in range(params.kfold_num):
            train_k = shuffle(label.loc[label['source']=='chembl'], random_state = params.random_seed)
            valid_k = shuffle(label.loc[label['source']=='ncb'], random_state = params.random_seed)
            test_k = shuffle(label.loc[label['source']=='ncb'], random_state = params.random_seed)
            savecv_path = prj_path / 'data' / 'processed_data' / 'split_cvdata' / str(params.source.split(',')[0]+'+'+params.source.split(',')[1]) / f'{k}th_fold'
            savecv_path.mkdir(parents=True, exist_ok=True)
            train_k.to_csv(savecv_path / 'train_k.csv' )
            valid_k.to_csv(savecv_path / 'valid_k.csv' )
            test_k.to_csv(savecv_path / 'test_k.csv' )
    else: raise ValueError('data_source error')

def drugllmfea_from_local(drug, feature_dict=None, type='xmol'):
    class_factory = ['xmol_step_400000']
    bitsinfo = pd.read_csv(prj_path / 'data' / 'original_data' / f'{type}_bitsinfo_class.csv',index_col=None, header=0, low_memory=False)
    # bitsinfo.columns=['IDs', 'Subtypes']

    _df = bitsinfo.melt('IDs')
    _ds = _df.loc[_df['value'].eq(1)].groupby(by='IDs').apply(lambda x:','.join(x['variable']))
    _ds.name = 'Subtypes'
    bitsinfo=bitsinfo.join(_ds, on='IDs')

    colors = ['#FF6633']
    colors = sns.palettes.color_palette("PuBu_d", n_colors=len(class_factory)).as_hex()
    colormaps = dict(zip(class_factory, colors)) 
    colormaps.update({'NaN': '#000000'})

    if not feature_dict:
        flag = 'all'
        keys = class_factory
        cm = colormaps
    else:
        keys = [key for key in set(feature_dict.keys()) & set(class_factory)]
        flag = 'auto'
        cm = {}
        for k in class_factory:
            if k in keys:
                cm[k] = colormaps[k]

    drug_info = drug[bitsinfo['IDs']]

    return drug_info, bitsinfo, cm

def protllmfea_from_local(protein, feature_dict=None, type='esm2'):
    class_factory = ['esm2_t33_650M_UR50D','esm2_t36_3B_UR50D','esm2_t48_15B_UR50D']
    bitsinfo = pd.read_csv(prj_path / 'data' / 'original_data' / f'{type}_bitsinfo_class.csv',index_col=None, header=0, low_memory=False)
    # bitsinfo.columns=['IDs', 'Subtypes']

    _df = bitsinfo.melt('IDs')
    _ds = _df.loc[_df['value'].eq(1)].groupby(by='IDs').apply(lambda x:','.join(x['variable']))
    _ds.name = 'Subtypes'
    bitsinfo=bitsinfo.join(_ds, on='IDs')

    colors = ['#FF6633']
    colors = sns.palettes.color_palette("PuBu_d", n_colors=len(class_factory)).as_hex()
    colormaps = dict(zip(class_factory, colors)) 
    colormaps.update({'NaN': '#000000'})

    if not feature_dict:
        flag = 'all'
        keys = class_factory
        cm = colormaps
    else:
        keys = [key for key in set(feature_dict.keys()) & set(class_factory)]
        flag = 'auto'
        cm = {}
        for k in class_factory:
            if k in keys:
                cm[k] = colormaps[k]

    prot_info = protein[bitsinfo['IDs']]

    return prot_info, bitsinfo, cm

def to_dist_matrix(data, datatype, idx, tag, methods = ['correlation', 'cosine', 'jaccard']):
    df_dic = {}
    for method in methods:
        res = calculator.pairwise_distance(data, n_cpus=12, method=method)
        res = np.nan_to_num(res,copy=False)
        df = pd.DataFrame(res,index=idx,columns=idx)
        save_path = prj_path / 'data' / 'processed_data' / f'{datatype}' / 'scale'
        save_path.mkdir(parents=True, exist_ok=True)
        df.to_pickle(save_path / f'{tag}_{method}.cfg')
        df_dic[method] = df
    return df_dic

def drug_fea(drugs, type):
    print('...loading drug-llm-features...')
    drugs_fea, bitsinfo, colormaps_d = drugllmfea_from_local(drugs, feature_dict=None, type=type)
    print('drugs_fea.shape before integrated: ',drugs_fea.shape)
    return drugs_fea, bitsinfo

def protein_fea(proteins, type):
    print('...loading protein-llm-features...')
    prots_fea, bitsinfo, colormaps_p = protllmfea_from_local(proteins, feature_dict=None, type=type)
    print('prots_fea.shape before integrated: ',prots_fea.shape)
    return prots_fea, bitsinfo


def fea_statistic(fea):
    S = summary.Summary(n_jobs = 10)
    res= []
    for i in tqdm.tqdm(range(fea.shape[1])):
        r = S._statistics_one(fea.values, i)
        res.append(r)
        
    df = pd.DataFrame(res)
    df.index = fea.columns
    return df

def factor_int(n):
    val = math.ceil(math.sqrt(n))
    val2 = int(n/val)
    while val2 * val < float(n):
        val2 += 1
    return val, val2