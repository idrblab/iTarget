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

from feamap.preparation import relationship_data, split_data

prj_path = Path(__file__).parent.resolve()
save_path = prj_path / 'data' / 'processed_data'
save_path.mkdir(parents=True, exist_ok=True)
save_path_df = prj_path / 'data' / 'processed_data' / 'drug_fea' / 'map_transferred'
save_path_df.mkdir(parents=True, exist_ok=True)
save_path_cf = prj_path / 'data' / 'processed_data' / 'protein_fea' / 'map_transferred'
save_path_cf.mkdir(parents=True, exist_ok=True)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--split_data", type=str, default='False', choices=['True', 'False'], help="")
    parser.add_argument("--kfold_cv_type", type=str, default='pair', choices=['pair', 'drug', 'protein'], help="K-Fold Cross-Validation dataset splitting choice")
    parser.add_argument("--data_wash", action='store_true', help="whether to wash the data")
    parser.add_argument("--sampling", type=str, default='NONE', choices=['DOWN', 'NONE'], help="")
    parser.add_argument("--kfold_num", type=int, default=5, help="number of folds for K-Fold Cross-Validation, default 5")
    parser.add_argument("--source", type=str, default='example', help="the data source that the task working on")
    params = parser.parse_args()
    print(vars(params))


    # split train, valid and test
    if (params.split_data == 'True') & (params.source != 'bindingdb'):
        label = relationship_data(params.source)
        split_data(params, label, )

