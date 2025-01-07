import ironmask
import torch
import math
from tqdm.auto import tqdm
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
import train_TE




def Run(_embedding_path:str, _bin_path:str, _title:str, _expand_dim=512, _nonzero=16):
#     lfw_embeddings=torch.load('./features/lfw_1024.pt')
    loaded_embedding=torch.load(_embedding_path)
    
#     lfw_issame_list=load_issame("./features/lfw.bin")
    loaded_issame_list=load_issame(_bin_path)
    dimension_available = [512,1024,2048,4096,8192]
    if _expand_dim not in dimension_available:
        print(f"Input dimension should be among {dimension_available}")
        return
    if _nonzero<10 or _nonzero>16:
        print("10<= non_zero <=16")
        return
    RunModel(expand_dim=_expand_dim,
                     nonzero=_nonzero,
                     embeddings=loaded_embedding,
                     issame_list=loaded_issame_list,
                     title=f'{_title}-{_expand_dim}-{_nonzero}')
    
    
    

def RunModel(expand_dim,nonzero,embeddings,issame_list,title):
    #embeddings_L=embeddings[0::2]
    #embeddings_R=embeddings[1::2]
    result=torch.empty(len(embeddings)//2)
    
    semi_orthogonal=torch.empty(512,expand_dim*2)
    semi_orthogonal=torch.nn.init.orthogonal_(semi_orthogonal)
    TE=train_TE.training_nn(expand_dim*2,expand_dim)
    embeddings_L=torch.mm(embeddings[0::2],semi_orthogonal)
    embeddings_R=torch.mm(embeddings[1::2],semi_orthogonal)
    
    
    for i in tqdm(range(len(issame_list))):
        
        
#         enroll_t = embeddings_L[i].reshape(-1,1)
#         verify_t_prime = embeddings_R[i].reshape(-1,1)
        enroll_t = TE(embeddings_L[i].reshape(-1,1)).reshape(-1,1)
        verify_t_prime = TE(embeddings_R[i].reshape(-1,1)).reshape(-1,1)



        r,P=ironmask.GEN_N(enroll_t,n=expand_dim,nonzero=nonzero)
        r_prime=ironmask.REP_N(P,verify_t_prime,nonzero=nonzero)

        if r==r_prime and issame_list[i]==True:
            result[i]=True        
        elif r!=r_prime and issame_list[i]==False:
            result[i]=True
        else:
            result[i]=False

        if (i%100)==0 and i!=0:
            print("real-time accuracy : "+str((100*result[:i].sum()/i).item())+" %")
            if np.invert(issame_list[:i]).sum()!=0:
                print("real-time FAR : "+str(100*(1-result[:i][list(np.invert(issame_list[:i]))].sum()/result[:i][list(np.invert(issame_list[:i]))].size(0)).item())+" %")
            print("real-time FRR : "+str(100*(1-result[:i][issame_list[:i]].sum()/result[:i][issame_list[:i]].size(0)).item())+" %")

    datas = [str((100*result.sum()/(len(embeddings)//2)).item())+" %",
        str(100*(1-result[list(np.invert(issame_list))].sum()/result[list(np.invert(issame_list))].size(0)).item())+" %",
        str(100*(1-result[issame_list].sum()/result[issame_list].size(0)).item())+" %"]
    df_creator(datas).to_csv(f'./{title}.csv')  
    
    print("Accuracy : "+datas[0])
    print("FAR : "+datas[1])
    print("FRR : "+datas[2])

    print("  ")
    print("====================")
    print("  ")
    
    
    
def load_issame(path):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3
    
    return issame_list

def df_creator(data):
    return pd.DataFrame(data=[data], columns= ['Accuracy','FAR','FRR'])