import gc
import argparse
import os
import psutil
import pandas as pd
import numpy as np
import csv
import torch
import esm
from pathlib import Path
import sys
prj_path = Path(__file__).parent.resolve()
sys.path.append(prj_path)

# esm2type = 'esm2_t33_650M_UR50D'
# repr_layer = 33
# hid_dim = 1280
# esm2type = 'esm2_t36_3B_UR50D'
# repr_layer = 36
# hid_dim = 2560
# esm2type = 'esm2_t48_15B_UR50D'
# repr_layer = 48
# hid_dim = 5120

def run(start_index,esm2type,datatype,repr_layer,hid_dim):
    # Load ESM-2 model
    # model_location = str(f'/home/minjie/home/downloads/esm/{esm2type}.pt')
    model_location = str(prj_path / 'pretrained_esm2_models' / f'{esm2type}.pt')
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_location)
    # model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)


    _data = pd.read_csv(prj_path / 'data' / f'{datatype}_prots.csv')
    _data['length'] = _data['sequence'].str.len()
    _data = _data.sort_values(by=['length'], )


    data = _data[['protid','sequence']].apply(lambda x: tuple(x), axis=1).values.tolist()
    head_flag = False
    resume_flag = False
    columns = [f'{esm2type}_idx{i}' for i in range(hid_dim)]
    columns.insert(0,'protid')

    save_path = prj_path / 'data' / f'{esm2type}' / f'{datatype}' / 'token_representations'
    save_path.mkdir(parents=True, exist_ok=True)

    _data.to_csv(prj_path / 'data' / f'{esm2type}' / f'{datatype}' / f'{datatype}_prots_sorted.csv')

    for index,row in data:
        # print(f'{esm2type},{repr_layer},{hid_dim}')
        # data = [
        #     ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        #     ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        #     ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        #     ("protein3",  "K A <mask> I S Q"),
        # ]
        
        if start_index=='head':
            resume_flag=True
        elif start_index==index: # once touch the target id, keep the resume procedue on
            resume_flag=True
            head_flag=True

        if resume_flag:
            batch_labels, batch_strs, batch_tokens = batch_converter([(index,row)])
            # batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            # Extract per-residue representations (on CPU)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=True)
            token_representations = results["representations"][repr_layer]

            # Generate per-sequence representations via averaging
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            sequence_representations = []
            # for i, tokens_len in enumerate(batch_lens):
            for i, (tokens_lab, tokens_len) in enumerate(zip(batch_labels,batch_lens)):
                # sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
                with open(save_path / f'{tokens_lab}.npy', "wb") as f:
                    np.save(f,token_representations[i, 1 : tokens_len - 1].numpy())
                     
            del token_representations
            gc.collect()

            for i, tokens_lab in enumerate(batch_labels):
                sequence_representations.append(np.load(save_path / f'{tokens_lab}.npy').mean(0))
                gc.collect()


            # pd.DataFrame(sequence_representations, index=_data.protid, columns=columns).to_csv(save_path / 'all-data-merge-prot.csv')
            with open(prj_path / 'data' / f'{esm2type}' / f'{datatype}' / f'{datatype}_all-data-merge-prot.csv', "a+") as f:
                writer = csv.writer(f)
                for idx, (lab, len, rep) in enumerate(zip(batch_labels,batch_lens,sequence_representations)):
                    if not head_flag:
                        writer.writerow(columns)
                        head_flag = True
                    result = [lab]
                    for r in rep:
                        result.append(r)
                    writer.writerow(result)
            
            print ('当前进程的运行对象：',index,'+',batch_lens)
            print ('当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
            del sequence_representations
            gc.collect()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", type=str, default='head')
    parser.add_argument("--esm2type", type=str, default='esm2_t36_3B_UR50D')
    parser.add_argument("--datatype", type=str, default='human')
    parser.add_argument("--repr_layer", type=int, default=33)
    parser.add_argument("--hid_dim", type=int, default=1280)
    
    params = parser.parse_args()
    print(vars(params))

    # esm2type = 'esm2_t33_650M_UR50D'
    # repr_layer = 33
    # hid_dim = 1280
    # esm2type = 'esm2_t36_3B_UR50D'
    # repr_layer = 36
    # hid_dim = 2560
    # esm2type = 'esm2_t48_15B_UR50D'
    # repr_layer = 48
    # hid_dim = 5120
    run(params.start_index,params.esm2type,params.datatype,params.repr_layer,params.hid_dim)