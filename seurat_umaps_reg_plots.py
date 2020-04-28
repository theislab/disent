import numpy as np
import scanpy as sc
import glob, os
import matplotlib
import pandas as pd
import beta_vae_5
import shutil
import re
from reg_plot import *


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
    os.chdir(path)
    actual_data_temp = actual_data[actual_data.obs["cell_type"]==clust_typ]
    reg_mean_vals = []
    for file in glob.glob("*.h5ad"):
        print(file)
        adata = sc.read(file)
        
        pred_data = sc.AnnData(adata.X, obs={"comparison_typ":["pred"]*len(adata)}, var={"var_names":adata.var_names})
        actual_data_temp = sc.AnnData(actual_data_temp.X, obs={"comparison_typ":["actual"]*len(actual_data_temp)}, var={"var_names":actual_data_temp.var_names})
        
        first_cell = file[0:file.find('.')]
        #print(first_cell)
        
        plot_data = actual_data_temp.concatenate(pred_data)
        
        top_100_gene_list = list(actual_data.uns["rank_genes_groups"]['names'][clust_typ])
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
        
        if reg_val[1] >= 0.70:
            source_cell = [string for string in cells if string in file]
            source_cell = source_cell[0]
            generate_simulated_umaps(path,actual_data,clust_typ,source_cell,adata,first_cell)
    return reg_mean_vals

def main():
    
    #os.chdir("/storage/groups/ml01/workspace/harshita.agarwala/")
    path_folder = r'./models_seurat_leave_one'
    #path_folder = r'./models_seurat_hcv_500epochs_try'
    cells = list(set(adata.obs["cell_type"]))
    folders = os.listdir(path=path_folder)
    for folder_name in folders:
        print(folder_name)
        '''
        if os.path.exists(path_folder+"/"+folder_name+"/gene_heatmaps/figures"):
            print("path exists")
            try:
                shutil.rmtree(path_folder+"/"+folder_name+"/gene_heatmaps/figures")
                print("deleted folder")
            except e as Exception:
                print(e)
                pass
        '''
        try:
            cell_to_drop = [string for string in cells if string in folder_name]
            print(cell_to_drop[0])
            cell_to_drop = max(cell_to_drop, key=len)
            df_list = generate_simulated_reg_plots(path=path_folder+"/"+folder_name+"/gene_heatmaps/",
                                        actual_data=adata,clust_typ = cell_to_drop,cells=cells)
            df = pd.DataFrame(df_list,columns=["name","r_sq_all","r_sq_100"])
            os.chdir("/home/icb/harshita.agarwala/")
            #os.chdir("/storage/groups/ml01/workspace/harshita.agarwala/")
            df.to_csv(path_folder+"/"+folder_name+"/gene_heatmaps/reg_mean.csv",index=False)
        except:
            pass

if __name__ == "__main__":  
    main()

