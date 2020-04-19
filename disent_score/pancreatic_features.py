import warnings
import os
warnings.filterwarnings('ignore')
import beta_vae
import anndata
import scanpy as sc
import pandas as pd
import numpy as np
import os
from random import shuffle
import random

import numpy as np
import scanpy as sc
from scipy import sparse
from sklearn import preprocessing

def shuffle_adata(adata):
    """
        Shuffles the `adata`.
        # Parameters
        adata: `~anndata.AnnData`
            Annotated data matrix.
        labels: numpy nd-array
            list of encoded labels
        # Returns
            adata: `~anndata.AnnData`
                Shuffled annotated data matrix.
            labels: numpy nd-array
                Array of shuffled labels if `labels` is not None.
        # Example
        ```python
        import scgen
        import anndata
        import pandas as pd
        train_data = anndata.read("./data/train.h5ad")
        train_labels = pd.read_csv("./data/train_labels.csv", header=None)
        train_data, train_labels = shuffle_data(train_data, train_labels)
        ```
    """
    if sparse.issparse(adata.X):
        adata.X = adata.X.A

    ind_list = [i for i in range(adata.shape[0])]
    shuffle(ind_list)
    new_adata = adata[ind_list, :]
    return new_adata

def feature_scores(model,L,B,data):
    z_diff_list_cluster = []
    z_diff_list_phase = []
    z_diff_list_depth = []
    z_diff_list_exp = []
    #cell_names = list(set(data.obs["cell_type"]))

    for batch in range(B):
        print("Batch no: ",str(batch))
        average_z_diff_cluster = 0
        average_z_diff_phase = 0
        average_z_diff_depth = 0
        average_z_diff_exp = 0
        
        data = shuffle_adata(data)
        sampled_data = data[0:L,:]
        remaining_data = data[L:,:]
        print(remaining_data)
        for l in range(L):
            try:
                #print(l)
                first_sample = sampled_data[l,:]
                cluster_type = first_sample.obs["clusters"][0]
                phase_type = first_sample.obs["cell_phase"][0]
                depth_type = first_sample.obs["seq_depth"][0]
                exp_type = first_sample.obs["exp_gene"][0]

                #print(cluster_type,phase_type,depth_type,exp_type)

                first_sample = first_sample.X
                first_sample = np.reshape(first_sample,(1,data.shape[1]))
                #print(first_sample)
                z_1 = model.to_latent(first_sample)
                #print("z_1",z_1)

                remaining_sample = remaining_data[remaining_data.obs["clusters"]==cluster_type]
                rand = random.randrange(0,len(remaining_sample))
                second_sample_cluster = remaining_sample[rand,:]
                second_sample_cluster = np.reshape(second_sample_cluster.X,(1,data.shape[1]))
                #print(second_sample_cluster)
                z_2 = model.to_latent(second_sample_cluster)
                #print("z_2",model.to_latent(second_sample_cluster))

                remaining_sample_1 = remaining_data[remaining_data.obs["cell_phase"]==phase_type]
                rand_1 = random.randrange(0,len(remaining_sample_1))
                second_sample_phase = remaining_sample_1[rand_1,:]
                second_sample_phase = np.reshape(second_sample_phase.X,(1,data.shape[1]))
                z_3 = model.to_latent(second_sample_phase)
                #print("z_3",z_3)

                remaining_sample_2 = remaining_data[remaining_data.obs["seq_depth"]==depth_type]
                print("same seq data",remaining_sample_2)
                rand_2 = random.randrange(0,len(remaining_sample_2))
                second_sample_depth = remaining_sample_2[rand_2,:]
                second_sample_depth = np.reshape(second_sample_depth.X,(1,data.shape[1]))
                z_4 = model.to_latent(second_sample_depth)

                remaining_sample_3 = remaining_data[remaining_data.obs["exp_gene"]==exp_type]
                rand_3 = random.randrange(0,len(remaining_sample_3))
                second_sample_exp = remaining_sample_3[rand_3,:]
                second_sample_exp = np.reshape(second_sample_exp.X,(1,data.shape[1]))
                z_5 = model.to_latent(second_sample_exp)

                print("z_1",z_1)
                print("z_2",z_2)
                z_diff_cluster = abs(z_1[0,:]-z_2[0,:])
                #print("diff_Cell",z_diff_cell)
                average_z_diff_cluster = average_z_diff_cluster + z_diff_cluster

                z_diff_phase = abs(z_1[0,:]-z_3[0,:])
                average_z_diff_phase = average_z_diff_phase + z_diff_phase

                z_diff_depth = abs(z_1[0,:]-z_4[0,:])
                average_z_diff_depth = average_z_diff_depth + z_diff_depth

                z_diff_exp = abs(z_1[0,:]-z_5[0,:])
                average_z_diff_exp = average_z_diff_exp + z_diff_exp
		
                print("avg_diff",average_z_diff_cluster)
                print("avg_diff",average_z_diff_phase)
                print("avg_diff",average_z_diff_depth)
            except Exception as e:
                print(e)
                pass
        average_z_diff_cluster = average_z_diff_cluster/L
        average_z_diff_phase = average_z_diff_phase/L
        average_z_diff_depth = average_z_diff_depth/L
        average_z_diff_exp = average_z_diff_exp/L

        z_diff_list_cluster.append([list(average_z_diff_cluster)])
        z_diff_list_phase.append([list(average_z_diff_phase)])
        z_diff_list_depth.append([list(average_z_diff_depth)])
        z_diff_list_exp.append([list(average_z_diff_exp)])
        
    df_cluster = pd.DataFrame(data={"y": ["cluster"]*len(z_diff_list_cluster), "avg_z_diff": z_diff_list_cluster})
    df_phase = pd.DataFrame(data={"y": ["cell_phase"]*len(z_diff_list_phase), "avg_z_diff": z_diff_list_phase})
    df_depth = pd.DataFrame(data={"y": ["seq_depth"]*len(z_diff_list_depth), "avg_z_diff": z_diff_list_depth})
    df_exp = pd.DataFrame(data={"y": ["exp_gene"]*len(z_diff_list_exp), "avg_z_diff": z_diff_list_exp})

    df = pd.concat([df_cluster,df_phase,df_depth,df_exp])
    return df    