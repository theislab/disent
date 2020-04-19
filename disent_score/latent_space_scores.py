import warnings
import os
warnings.filterwarnings('ignore')
import beta_vae
import util_lossC as ul
import scanpy as sc
import pandas as pd
import numpy as np

import random

def observation_scores(L,B,data):
    Y = data.iloc[:,0]
    obs_feat = list(set(Y))

    dataframe = pd.DataFrame(columns = ["y","avg_z"])
    for obs in obs_feat:
        df = general_feat_scores(L,B,data,obs)
        dataframe = pd.concat([dataframe,df])
    return dataframe

def general_feat_scores(L,B,data,obs):
    z_list = []
    for batch in range(B):
        #print(obs)
        try:
            sampled_data = data[data.iloc[:,0]==obs]
            sampled_data = sampled_data.drop([sampled_data.columns[0]] ,  axis='columns')
            sampled_data = sampled_data.sample(n=L, random_state=1+batch)
            average_z = np.mean(sampled_data,axis=0)
        except:
            sampled_data = data[data.iloc[:,0]==obs]
            sampled_data = sampled_data.drop([sampled_data.columns[0]] ,  axis='columns')
            average_z = np.mean(sampled_data,axis=0)
        z_list.append([list(average_z)])
        

    df = pd.DataFrame(data={"y": obs, "avg_z": z_list})
    return df