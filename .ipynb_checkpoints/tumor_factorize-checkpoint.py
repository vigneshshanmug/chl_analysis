import scanpy as sc
import pyro
import torch
import warnings

import pandas as pd
import seaborn as sns
from tqdm import trange
import numpy as np
import anndata as ad
import ipywidgets as widgets
import matplotlib.pyplot as plt
from tensorly.cp_tensor import CPTensor

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor


from scBTF import SingleCellTensor, SingleCellBTF, FactorizationSet, Factorization, BayesianCP

sc.logging.print_header()
sc.settings.njobs = 32


adata = sc.read_h5ad('../final/data/091222_combined_dataset_final_v2.h5ad')
adata = adata[adata.obs.cell_types_level_4 == "Tumor"]

print(adata.n_obs, adata.n_vars)
print(adata.X.expm1().sum(axis=1).round()[:5])
adata.raw = adata



tensor = torch.tensor((adata.X.expm1().todense()*100).round())
print(tensor.shape)

consensus = False
n_restarts = 2
num_steps = 1500

for rank in [11,12,13,14]:
    print(f"Processing rank {rank}")
    if consensus:
        factorization_set = []
        for i in trange(n_restarts):
            bayesianCP = BayesianCP(dims=tensor.shape, rank=rank, init_alpha=1000.0)
            bayesianCP.fit(tensor, num_steps=num_steps, progress_bar=True)
            precis = bayesianCP.precis(tensor)
            factorization_set.append(CPTensor((torch.ones(rank), [precis[f'factor_{i}']['mean'] for i in range(2)])))

        print("Factorization done.")
        data = np.column_stack([factorization.factors[1].numpy() for factorization in factorization_set]).T
        sums = data.T.sum(axis=0)
        sums[sums == 0] = 1
        data_normed = (data.T / sums).T * 1e5
        
        print(f"Processing generating consensus for {rank}")
        
        kmeans = KMeans(init="k-means++", n_clusters=rank, n_init=20, tol=1e-8)
        labels_ = make_pipeline(StandardScaler(), kmeans).fit(data_normed)[-1].labels_
        lof = LocalOutlierFactor(n_neighbors=50, contamination=0.002, metric='euclidean')
        outliers = make_pipeline(StandardScaler(), lof).fit_predict(data_normed)

        medians = np.stack(
            np.median(data_normed[(labels_ == group) & (outliers != -1), :], axis=0) for group in np.unique(labels_))
        bayesianCP = BayesianCP(dims=tensor.shape, rank=rank, init_alpha=10000.0, init_beta=0.3, model='zero_inflated_poisson_fixed',
                                fixed_mode=1, fixed_value=torch.from_numpy(medians.T).float())
        bayesianCP.fit(tensor, num_steps=num_steps)
        prs = bayesianCP.precis(tensor)
        a, b = [prs[f'factor_{i}']['mean'] for i in range(2)]
    else:
        bayesianCP = BayesianCP(dims=tensor.shape, rank=rank, init_alpha=1e3, init_beta=0.1, model='zero_inflated_poisson_auto')
        svi = bayesianCP.fit(tensor, num_steps = num_steps, progress_bar=True)
        prs = bayesianCP.precis(tensor, num_samples=10)
        print("here")
        a, b = [prs[f'factor_{i}']['mean'] for i in range(2)]

        tensor_means = torch.einsum('ir,jr->ij', a, b)


        print(1 - (torch.norm(tensor.float() - tensor_means.float())**2 / torch.norm(tensor.float())**2))

    cell_components = pd.DataFrame(a, columns=[f'Factor_{i}' for i in range(rank)], index=adata.obs.index)
    cell_components = pd.concat([adata.obs, cell_components], axis=1)
    cell_components.to_csv(f'cell_components_rank_{rank}.csv')

    gene_complonents = pd.DataFrame(b, columns=[f'Factor_{i}' for i in range(rank)], index=adata.var.index)
    gene_complonents.to_csv(f'gene_components_rank_{rank}.csv')
