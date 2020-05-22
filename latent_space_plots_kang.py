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

sns.set(font_scale=1)
sns.set_style("darkgrid")

def double_feature_to_latent(path,adata,feature,feature2,model,z_dim):

    old_path = os.getcwd()

    cell_in_latentspace = model.to_latent(adata.X)
    df_cols = []
    
    for i in range(z_dim):
        df_cols.append(str(i)+'dim')
    
    #print(df_cols)

    latent_df = pd.DataFrame(cell_in_latentspace,index=adata.obs[feature],
                             columns=df_cols)
    latent_df.reset_index(level=0, inplace=True)
    latent_df[feature2] = list(adata.obs[feature2])
    print(latent_df)
    
    path = path+"cells_latent_"+feature+feature2+"/"
    
    try:
        os.makedirs(path)
    except OSError:
        print ("Check if path %s already exists" % path)
    else:
        print ("Successfully created the directory %s" % path)
    
    os.chdir(path)
    latent_df.to_csv("cells_in_latent.csv")
    
    for i in range(z_dim):
        dim_col = str(i)+"dim"
        latent_df["groups_dim"] = round(latent_df[dim_col],1)
        dim0_count = latent_df.groupby(["groups_dim",feature,feature2]).count()
        dim0_count = dim0_count.reset_index(level=[0,1,2])
        dim0_count = dim0_count.loc[:,[dim_col,"groups_dim",feature,feature2]]
        
        print(dim0_count)

        fig, ax = plt.subplots(figsize=(6,6))
        scatter = sns.scatterplot(dim0_count["groups_dim"],dim0_count[feature],
                   size=dim0_count[dim_col].values,hue=dim0_count[feature2],linewidth=0,
                                  sizes=(10, 150))
        
        
        scatter.set_title("Latent Space for Dimension "+str(i+1), weight="bold")
        scatter.set_ylabel(feature.capitalize())
        scatter.set_xlabel("Linear scale")
        plt.show()

        plt.savefig(dim_col+".png", bbox_inches='tight',dpi=100)
    os.chdir(old_path)