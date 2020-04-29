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

sns.set(font_scale=1)
sns.set_style("darkgrid")

def plot_kl_loss(path,z_dim):
    df = pd.read_csv(path+"csv_logger.log")
    kl_cols = []
    for i in range(z_dim):
        df.rename(columns={"kl_loss_monitor"+str(i): "dimension"+str(i+1)},inplace=True)
        kl_cols.append("dimension"+str(i+1))
    #print(df)
    df = df[kl_cols]
    #print(df)
    #print(df)
    df.plot()
    plt.title("KL loss per dimension over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("KL Loss")
    plt.savefig(path+"kl_plot.png",dpi=150)
