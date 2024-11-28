#### prepare pretrained models in `./pretrained_esm2_models/` according to `download_url.txt`, and protein sequences according to files in `./data/uniprot_prots.csv` or `./data/template_prots.csv`
#### Then run
```
conda activate esm2
cd ./bashes
sh template_esm2_t36_3B_UR50D.sh	# {--datatype}='uniprot' for using 'uniprot_prots.csv' file in ./data, ='template' for using 'template_prots.csv' file in ./data
```

