# iTarget





## Dependencies

- should be deployed on Linux in python 3.8.
- Main requirements: `python==3.8.8`, `pytorch==1.8.1`.
- To use GPU, please install the GPU version of  `pytorch`.



## Install

1. Download source codes.
2. Should be deployed on Linux.
3. Python environment preparation

   We provide three packed conda environments for users to construct Python dependencies using Anaconda.

```
# operate in your own conda envs path, usullaly, in `~/anaconda3/envs` by default.
mkdir ~/anaconda3/envs/xmol
tar -zxvf ./_conda_envs/xmol.tar.gz -C ~/anaconda3/envs/xmol
mkdir ~/anaconda3/envs/esm2
tar -zxvf ./_conda_envs/esm2.tar.gz -C ~/anaconda3/envs/esm2
mkdir ~/anaconda3/envs/iTarget
tar -zxvf ./_conda_envs/iTarget.tar.gz -C ~/anaconda3/envs/iTarget 
```



## Usage

#### 1. Prepare LLM representation for proteins and compounds using Large Language Models (ESM-2 and X-MOL in this study)

##### 1.1 Preprocess data for benchmarks. 

```
python _data_preprocess.py
# the produced '{type}_drugs.csv' and '{type}_prots.csv' files could be used in step 1.2
```

##### 1.2 Switch to the target path and work following the tutorials in the corresponding file. 

```
cd ./_ForFeatures/esm2/		# for proteins
cd ./_ForFeatures/xmol/		# for compounds

# after finishing representaion, back to the project root path
```

#### 2. Template map construction and Feature map transformation

##### 2.1 For template maps, move the produced LLM feature files in step 1.2 to the working path `./data/original_data/scale/`. 

```
mv ./_ForFeatures/esm2/data/{--esm2type}/{--datatype}/{--datatype}_all-data-merge-prot.csv ./data/original_data/scale/
# for proteins' template, by default, {--esm2type}='esm2_t36_3B_UR50D', {--datatype}='uniprot'

mv ./_ForFeatures/xmol/FT_to_embedding/data/for_output/{--datatype}_all-data-merge-drug.csv ./data/original_data/scale/
# for compounds' template, by default, {--datatype}='fullchembl'
```

##### 2.2 The moved feature files in `./data/original_data/scale/`  should be renamed using same {--scale_source} for {--datatype} according to the corresponding settings in downstream file `0_feadist.sh`. Here, we use 'uniprot+fullchembl' as an example, and then result in `uniprot+fullchembl_all-data-merge-prot.csv` and `uniprot+fullchembl_all-data-merge-drug.csv` two files in `./data/original_data/scale/`.

##### 2.3 For feature maps, move the produced LLM feature files to the working path `./data/original_data/`.

```
mv ./_ForFeatures/esm2/data/{--esm2type}/{--datatype}/{--datatype}_all-data-merge-prot.csv ./data/original_data/
# for proteins' features, by default, {--esm2type}='esm2_t36_3B_UR50D', {--datatype}='example' or user-defined

mv ./_ForFeatures/xmol/FT_to_embedding/data/for_output/{--datatype}_all-data-merge-drug.csv ./data/original_data/
# for compounds' features, by default, {--datatype}='example' or user-defined
```

##### 2.4 Switch to the bashes path for feature distance calculation and feature map transformation.

```
cd bashes
conda activate iTarget

# calculate feature distance
sh 0_feadist.sh	# by default, {--scale_method}='standard', {--scale_source}='uniprot+fullchembl'

# copy calculated configs to work path
cp ../data/processed_data/drug_fea/scale/standard/*.cfg ./feamap/config/trans_from_{--scale_source}/
cp ../data/processed_data/protein_fea/scale/standard/*.cfg ./feamap/config/trans_from_{--scale_source}/

# feature transformation
sh 1_trans_drug.sh	# for compounds, by default, {--scale_method}='standard', {--disttype}='uniprot+fullchembl', {--source}='example' or user-defined
sh 1_trans_prot.sh	# for proteins, by default, {--scale_method}='standard', {--disttype}='uniprot+fullchembl', {--source}='example' or user-defined
```

#### 3. Model training and Cross-validation

##### 3.1 Prepare dataset for cross-validation :

```
sh 2_split_cvdata.sh
# optional, or you can prepare files following the examples in `./data/processed_data/split_cvdata/`.
# This step is not required for bindingdb benchmark, has done in step 1.1
```

##### 3.2 Run model training and cross-validation:

```
sh 3_train_cv.sh # by defalut, {--kfold_num}=5, {--task}='cv', {--n_epochs}=128, {--gpu}=0, {--batch_size}=512, {--lr}=5e-4, {--monitor}='auc_val', {--source}='example'
```



## Citation and Disclaimer

The manuscript is currently under peer review. Should you have any questions, please contact Dr. Zhang at hanyu_zhang@zju.edu.cn
