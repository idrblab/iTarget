import pandas as pd
import numpy as np
import csv
from pathlib import Path
import os 

prj_path = Path(__file__).parent.resolve()
# datatype = 'template'
datatype = 'fullchembl'
hidden = 768 # xmol

print('data type: {}'.format(datatype))

cpi = pd.read_csv(str(datatype)+"_drugs.csv")
head_flag = False


with open(str(datatype)+"_all-data-merge-drug.csv", "a+") as f:
    writer = csv.writer(f)

    columns=['xmol_idx'+str(i) for i in range(hidden)]
    columns.insert(0,'drugid')

    for idx, (drugid,smiles) in cpi.iterrows():
        if not head_flag:
            writer.writerow(columns)
            head_flag = True
        
        # drug_representation = np.array(pd.read_pickle(os.path.join(str(prj_path/str(datatype)/str(drugid)), str(drugid)+".pickle")).embed[0])
        # sequence_embed = drug_representation.mean(axis=0)
        sequence_embed = np.load(os.path.join(str(prj_path/str(datatype)/str(drugid)), str(drugid)+".npy"))
        # drugid = 'DR'+str(molid).zfill(6)
        result = [drugid]
        for r in sequence_embed:
            result.append(r)
        writer.writerow(result)


# drug_dict = {}
# for idx, (drugid,smiles) in cpi.iterrows():
#     drug_representation = np.array(pd.read_pickle(os.path.join(str(prj_path/str(datatype)/str(drugid)), str(drugid)+".pickle")).embed[0])
#     np.save(os.path.join(str(prj_path/str(datatype)/str(drugid)), str(drugid)+".npy"), drug_representation)
#     drug_dict[drugid]=drug_representation
# import pickle
# with open(os.path.join(str(prj_path),str(datatype)+"_drugs.pickle"), 'wb') as f:
#     pickle.dump(drug_dict,f)