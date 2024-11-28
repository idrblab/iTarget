from feamap.utils.logtools import print_info
import sys
import gdown
import pandas as pd
import os
from pathlib import Path

prj_path = Path(__file__).parent.resolve().parent.resolve().parent.resolve()

def load_config(disttype = 'ALL', datatype = 'protein_fea', ftype = 'esm2', metric = 'cosine'):
    print(prj_path)
    if metric=='scale':
        if datatype == 'drug_fea':
            df = pd.read_pickle(prj_path / 'feamap' / 'config' / f'trans_from_{disttype}' / f'{ftype}_{disttype}_scale.cfg')
        elif datatype == 'protein_fea':
            df = pd.read_pickle(prj_path / 'feamap' / 'config' / f'trans_from_{disttype}' / f'{ftype}_{disttype}_scale.cfg')
    else:
        try:
            df = pd.read_pickle(prj_path / 'feamap' / 'config' / f'trans_from_{disttype}' / f'{ftype}_{disttype}_{metric}.cfg')
        except:
            print('Error while loading feature distance matrix')
            sys.exit()
    return df
