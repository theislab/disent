
import warnings
warnings.filterwarnings('ignore')
import scanpy as sc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob, os
import matplotlib
import beta_vae_5
import seaborn as sns

sns.set(font_scale=1)
sns.set_style("darkgrid")

def single_feature_to_latent(path,adata,feature,model,z_dim):

    cell_in_latentspace = model.to_latent(adata.X)
    df_cols = []
    
    for i in range(z_dim):
        df_cols.append(str(i)+'dim')
    

    latent_df = pd.DataFrame(cell_in_latentspace,index=adata.obs[feature],
                             columns=df_cols)
    latent_df.reset_index(level=0, inplace=True)
    
    path = path+"cells_latent_"+feature+"/"
    
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
        dim0_count = latent_df.groupby(["groups_dim",feature]).count()
        dim0_count = dim0_count.reset_index(level=[0,1])
        dim0_count = dim0_count.loc[:,[dim_col,"groups_dim",feature]]
        
        #print(dim0_count)

        fig, ax = plt.subplots(figsize=(4,4))
        scatter = sns.scatterplot(dim0_count["groups_dim"],dim0_count[feature],
                   size=dim0_count[dim_col].values,linewidth=0)
        
        
        scatter.set_title("Latent Space for Dimension "+str(i+1), weight="bold")
        scatter.set_ylabel(feature.capitalize())
        scatter.set_xlabel("Linear scale")
        scatter.get_legend().set_title("Count")
        plt.show()

        plt.savefig(dim_col+".png", bbox_inches='tight',dpi=150)