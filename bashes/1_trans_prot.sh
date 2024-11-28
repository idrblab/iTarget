#!/usr/bin/bash

python -u ../1_map_transfer.py --disttype uniprot+fullchembl --source example --ftype esm2 --datatype protein_fea --channel False --fitmethod umap --metric cosine --scale_method standard
# cosine correlation jaccard 
# umap tsne mds
