#### prepare pretrained models in `./FT_to_embedding/data/model/step_400000` according to `download_url.txt`, and drug smiles according to files in `./FT_to_embedding/data/for_output/fullchembl_drugs.csv` or `./FT_to_embedding/data/for_output/template_drugs.csv`
```
conda activate xmol
```
#### Before using scripts, modify following files based on your needs
```
 |- ./FT_to_embedding/script/run_emb.sh  #  {--train_set}=${TASK_DATA_PATH}/{'fullchembl' for using 'fullchembl_drugs.csv' file, ='template' for using 'template_drugs.csv' file, or depending on your need}, {--test_set}=${TASK_DATA_PATH}/{'fullchembl' for using 'fullchembl_drugs.csv' file, ='template' for using 'template_drugs.csv' file, or depending on your need} \
 |- ./FT_to_embedding/data/for_output/pre_process.py  # datatype='fullchembl' for using 'fullchembl_drugs.csv' file, ='template' for using 'template_drugs.csv' file, or depending on your need
 |- ./FT_to_embedding/data/for_output/post_process.py  # datatype='fullchembl' for using 'fullchembl_drugs.csv' file, ='template' for using 'template_drugs.csv' file, or depending on your need
```
#### Then run:
```
cd ./FT_to_embedding/data/for_output/
python pre_process.py
cd ../../../
```
#### Then run:
```
cd ./bashes
sh template_xmol.sh
cd ../
```
#### Then run:
```
cd ./FT_to_embedding/data/for_output/
python post_process.py
cd ../../../
```
