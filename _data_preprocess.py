from pathlib import Path
import sys
prj_path = Path(__file__).parent.resolve()
sys.path.append(prj_path)
import pandas as pd

# type = 'human'
type = 'bindingdb'
# type ='biosnap'
# type = 'chembl'

human_cpi = pd.read_csv(prj_path / 'data' / '_original_data' / 'human_cpi' / 'human_cpi.txt', names=['smiles','sequence','label'], header=None, sep=" ")
bindingdb_cpi_train = pd.read_csv(prj_path / 'data' / '_original_data' / 'bindingdb_cpi' / 'train.txt', names=['smiles','sequence','label'], header=None, sep=" ")
bindingdb_cpi_dev = pd.read_csv(prj_path / 'data' / '_original_data' / 'bindingdb_cpi' / 'dev.txt', names=['smiles','sequence','label'], header=None, sep=" ")
bindingdb_cpi_test = pd.read_csv(prj_path / 'data' / '_original_data' / 'bindingdb_cpi' / 'test.txt', names=['smiles','sequence','label'], header=None, sep=" ")
biosnap_cpi = pd.read_csv(prj_path / 'data' / '_original_data' / 'biosnap_cpi' / 'full.csv')[['SMILES','Protein','Y']]
biosnap_cpi.rename(columns={'SMILES':'smiles','Protein':'sequence','Y':'label'}, inplace=True)
chembl_cpi = pd.read_csv(prj_path / 'data' / '_original_data' / 'chembl_cpi' / 'activity_with_seq_smiles_label_01_small.csv')[['compound_smiles','Sequence','label']]
chembl_cpi.rename(columns={'compound_smiles':'smiles','Sequence':'sequence','label':'label'}, inplace=True)


if type=='human':
    _cpi = human_cpi
elif type=='bindingdb':
    _cpi = pd.concat([bindingdb_cpi_train,bindingdb_cpi_dev,bindingdb_cpi_test])
elif type=='biosnap':
    _cpi = biosnap_cpi
elif type=='chembl':
    _cpi = chembl_cpi


DRs = _cpi['smiles'].drop_duplicates().reset_index(drop=True).to_frame()
DRs['drugid'] = 'DR'+DRs.index.to_series().apply(lambda x:str(x).zfill(6))
PRs = _cpi['sequence'].drop_duplicates().reset_index(drop=True).to_frame()
PRs['protid'] = 'PR'+PRs.index.to_series().apply(lambda x:str(x).zfill(6))

print(f'protein max length in {type}: ', PRs['sequence'].str.len().max())
print(f'drug max length in {type}: ', DRs['smiles'].str.len().max())

save_path = prj_path / 'data' / 'original_data'
save_path.mkdir(parents=True, exist_ok=True)

DRs[['drugid','smiles']].to_csv(prj_path / 'data' / '_original_data' / f'{type}_drugs.csv', index=False)
PRs[['protid','sequence']].to_csv(prj_path / 'data' / '_original_data' / f'{type}_prots.csv', index=False)
labels = _cpi.merge(DRs,on='smiles',how='left').merge(PRs,on='sequence',how='left')
labels['source']=[f'{type}'] * len(labels)
labels[['source','drugid','protid','label']].to_csv(save_path / f'{type}_prot-drug-index-source.csv', index=False)


if type=='bindingdb':
    label_train = bindingdb_cpi_train.merge(DRs,on='smiles',how='left').merge(PRs,on='sequence',how='left')
    label_train['source']=[f'{type}'] * len(label_train)
    label_dev = bindingdb_cpi_dev.merge(DRs,on='smiles',how='left').merge(PRs,on='sequence',how='left')
    label_dev['source']=[f'{type}'] * len(label_dev)
    label_test = bindingdb_cpi_test.merge(DRs,on='smiles',how='left').merge(PRs,on='sequence',how='left')
    label_test['source']=[f'{type}'] * len(label_test)

    save_path_cv = prj_path / 'data' / 'processed_data' / 'split_cvdata' / f'{type}'
    save_path_cv.mkdir(parents=True, exist_ok=True)

    label_train[['source','drugid','protid','label']].to_csv(save_path_cv / f'{type}_train-index.csv', index=True)
    label_dev[['source','drugid','protid','label']].to_csv(save_path_cv / f'{type}_dev-index.csv', index=True)
    label_test[['source','drugid','protid','label']].to_csv(save_path_cv / f'{type}_test-index.csv', index=True)


    