import numpy as np
import scanpy as sc
import glob, os
import matplotlib
import pandas as pd
import beta_vae
import shutil
import re
from reg_plot import *

def generate_simulated_pca(path,actual_data,clust_typ,source_cell,sim_data,first_cell):
    #print("in here")

    target = actual_data[actual_data.obs["cell_type"]==clust_typ]
    
    target = sc.AnnData(target.X, obs={"cell_type":["Target_"+ clust_typ]*len(target)}, 
                                       var={"var_names":target.var_names})
    
    source = actual_data[actual_data.obs["cell_type"]==source_cell]

    source = sc.AnnData(source.X, obs={"cell_type":["Source_"+source_cell]*len(source)}, 
                                   var={"var_names":source.var_names})
    predicted = sc.AnnData(sim_data.X, obs={"cell_type":["Predicted"]*len(sim_data)}, 
                                   var={"var_names":sim_data.var_names})
    
    combined_data = source.concatenate(target)
    combined_data = combined_data.concatenate(predicted)
    
    sc.pp.neighbors(combined_data)
    sc.tl.pca(combined_data, svd_solver='arpack')
    sc.pl.pca(combined_data, color=["cell_type"],
              legend_fontsize=12,
              palette = ['r','k','y'],
              save="_"+first_cell+"_to_"+clust_typ+"_celltypes.pdf")


def generate_simulated_umaps(path,actual_data,clust_typ,source_cell,sim_data,first_cell):
    target = actual_data[actual_data.obs["cell_type"]==clust_typ]
    target = sc.AnnData(target.X, obs={"cell_type":["Target_"+ clust_typ]*len(target)}, 
                                       var={"var_names":target.var_names})
    top_genes = list(actual_data.uns["rank_genes_groups"]['names'][clust_typ])

    source = actual_data[actual_data.obs["cell_type"]==source_cell]

    source = sc.AnnData(source.X, obs={"cell_type":["Source_"+source_cell]*len(source)}, 
                                   var={"var_names":source.var_names})
    predicted = sc.AnnData(sim_data.X, obs={"cell_type":["Predicted"]*len(sim_data)}, 
                                   var={"var_names":sim_data.var_names})

    combined_data = source.concatenate(target)
    combined_data = combined_data.concatenate(predicted)

    sc.pp.neighbors(combined_data)
    sc.tl.umap(combined_data)
    sc.pl.umap(combined_data, color=["cell_type"],
               legend_fontsize=12,
               save="_"+first_cell+"_to_"+clust_typ+"_celltypes.png",
               show=True,
               frameon=True,
                s = 35)
    sc.pl.umap(combined_data, color=top_genes[:3],
               legend_fontsize=12,
               save="_"+first_cell+"_to_"+clust_typ+"_top_genes.png",
               show=True,
               frameon=True,
                s = 35)

def generate_simulated_reg_plots(path,actual_data,clust_typ,cells):
    old_path = os.getcwd()
    os.chdir(path)
    print(os.getcwd())
    actual_data_temp = actual_data[actual_data.obs["cell_type"]==clust_typ]
    reg_mean_vals = []
    for file in glob.glob("*.h5ad"):
        print(file)
        adata = sc.read(file)
        print(adata)
        
        pred_data = sc.AnnData(adata.X, obs={"comparison_typ":["pred"]*len(adata)}, var={"var_names":adata.var_names})
        actual_data_temp = sc.AnnData(actual_data_temp.X, obs={"comparison_typ":["actual"]*len(actual_data_temp)}, var={"var_names":actual_data_temp.var_names})
        
        first_cell = file[0:file.find('.')]
        #print(first_cell)
        source_cell = [string for string in cells if string in file]
        source_cell = source_cell[0]
        
        plot_data = actual_data_temp.concatenate(pred_data)

        source_data = actual_data[actual_data.obs["cell_type"]==source_cell]

        source_data = sc.AnnData(source_data.X, obs={"cell_type":[source_cell]*len(source_data)}, 
                                   var={"var_names":source_data.var_names})
        target_data = sc.AnnData(actual_data_temp.X, obs={"cell_type":[clust_typ]*len(actual_data_temp)}, 
                                   var={"var_names":actual_data_temp.var_names})

        gene_data = target_data.concatenate(source_data)

        sc.tl.rank_genes_groups(gene_data, 'cell_type',method="t-test")
        top_100_gene_list = list(gene_data.uns["rank_genes_groups"]['names'][clust_typ])
        #print(top_100_gene_list)

        reg_val = reg_mean_plot(plot_data, condition_key="comparison_typ",
                                             axis_keys={"x": "actual", "y": "pred"},
                                             path_to_save="./reg_mean_"+file+"_TO_"+clust_typ+".png",
                                             legend=False,
                                             labels={"x": "actual", "y":"pred"},
                                             show=False,
                                             gene_list=top_100_gene_list[:5],
                                             top_100_genes = top_100_gene_list,
                                             fontsize=14,
                                             textsize=14)
        reg_mean_vals.append(list([first_cell,reg_val[0],reg_val[1]]))
        
        '''
        Threshold to decide if UMAPs and PCA plots should be plotted.
        '''
        if reg_val[1] >= 0.9:
            source_cell = [string for string in cells if string in file]
            source_cell = source_cell[0]
            #generate_simulated_umaps(path,actual_data,clust_typ,source_cell,adata,first_cell)
            generate_simulated_pca(path,actual_data,clust_typ,source_cell,adata,first_cell)
    os.chdir(old_path)
    return reg_mean_vals

