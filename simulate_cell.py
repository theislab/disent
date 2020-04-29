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
import beta_vae_5

def simulate_one_cell(path,data,cell,model,z_dim,feature):
    variable_names = data.var_names
    data_latent = model.to_latent(data.X)
    try:
        os.makedirs(path+"/gene_heatmaps/")
    except OSError:
        pass
    x_dim = data.shape[1]
    data_ast = data[data.obs[feature]==cell]
    cell_one = data_ast[0,:].X
    cell_one = np.reshape(cell_one,(1,x_dim))
    cell_one = model.to_latent(cell_one)

    for dim in range(z_dim):
        increment_range = np.arange(min(data_latent[:,dim]),max(data_latent[:,dim]),0.01)
        result_array = np.empty((0, x_dim))
        for inc in increment_range:
                cell_latent = cell_one
                #print(cell_latent)
                #print(cell_latent.shape)
                cell_latent[:,dim] = inc
                cell_recon = model.reconstruct(cell_latent)
                result_array = np.append(result_array,cell_recon,axis=0)

        result_adata = sc.AnnData(result_array, obs={"inc_vals":increment_range},var={"var_names":variable_names})
        result_adata.write(path+"/gene_heatmaps/"+str(cell)+"_"+str(dim)+".h5ad")

        
def simulate_multiple_cell(path,data,model,z_dim,feature):
    variable_names = data.var_names
    data_latent = model.to_latent(data.X)
    cells = list(set(data.obs["cell_type"]))
    try:
        os.makedirs(path+"/gene_heatmaps/")
    except OSError:
        pass
    x_dim = data.shape[1]
    
    for cell in cells:
        data_ast = data[data.obs[feature]==cell]
        cell_one = data_ast[0,:].X
        cell_one = np.reshape(cell_one,(1,x_dim))
        cell_one = model.to_latent(cell_one)

        for dim in range(z_dim):
            increment_range = np.arange(min(data_latent[:,dim]),max(data_latent[:,dim]),0.01)
            result_array = np.empty((0, x_dim))
            for inc in increment_range:
                    cell_latent = cell_one
                    #print(cell_latent)
                    #print(cell_latent.shape)
                    cell_latent[:,dim] = inc
                    cell_recon = model.reconstruct(cell_latent)
                    result_array = np.append(result_array,cell_recon,axis=0)

            result_adata = sc.AnnData(result_array, obs={"inc_vals":increment_range},var={"var_names":variable_names})
            result_adata.write(path+"/gene_heatmaps/"+str(cell)+"_"+str(dim)+".h5ad")
