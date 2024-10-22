import warnings
warnings.filterwarnings('ignore')
import scanpy as sc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import glob, os
import matplotlib
import re
import beta_vae

def simulate_one_cell(path,data,cell,model,z_dim,feature):
    variable_names = data.var_names
    data_latent = model.to_latent(data.X)
    latent_df = pd.DataFrame(data_latent)
    latent_df[feature] = list(data.obs[feature])
    try:
        os.makedirs(path+"/gene_heatmaps/")
    except OSError:
        pass
    x_dim = data.shape[1]
    data_ast = latent_df[latent_df[feature]==cell]
    cell_one = data_ast.iloc[[0],[0,1,2,3,4]]

    for dim in range(z_dim):
        increment_range = np.arange(min(data_latent[:,dim]),max(data_latent[:,dim]),0.01)
        result_array = np.empty((0, x_dim))
        for inc in increment_range:
                cell_latent = cell_one
                #print(cell_latent)
                #print(cell_latent.shape)
                cell_latent.iloc[:,dim] = inc
                cell_recon = model.reconstruct(cell_latent)
                result_array = np.append(result_array,cell_recon,axis=0)

        result_adata = sc.AnnData(result_array, obs={"inc_vals":increment_range},var={"var_names":variable_names})
        result_adata.write(path+"/gene_heatmaps/"+str(cell)+"_"+str(dim)+".h5ad")

        
def simulate_multiple_cell(path,data,model,z_dim,feature):
    variable_names = data.var_names
    data_latent = model.to_latent(data.X)
    latent_df = pd.DataFrame(data_latent)
    latent_df[feature] = list(data.obs[feature])
    cells = list(set(data.obs[feature]))
    try:
        os.makedirs(path+"/gene_heatmaps/")
    except OSError:
        pass
    x_dim = data.shape[1]
    
    for cell in cells:
        data_ast = latent_df[latent_df[feature]==cell]
        cell_one = data_ast.iloc[[0],[0,1,2,3,4]]

        for dim in range(z_dim):
            increment_range = np.arange(min(data_latent[:,dim]),max(data_latent[:,dim]),0.01)
            result_array = np.empty((0, x_dim))
            for inc in increment_range:
                    cell_latent = cell_one
                    #print(cell_latent)
                    #print(cell_latent.shape)
                    cell_latent.iloc[:,dim] = inc
                    cell_recon = model.reconstruct(cell_latent)
                    result_array = np.append(result_array,cell_recon,axis=0)

            result_adata = sc.AnnData(result_array, obs={"inc_vals":increment_range},var={"var_names":variable_names})
            result_adata.write(path+"/gene_heatmaps/"+str(cell)+"_"+str(dim)+".h5ad")
