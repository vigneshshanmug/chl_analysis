import scanpy as sc
import argparse
import numpy as np
import matplotlib.pyplot as plt

from scBTF import SingleCellTensor, SingleCellBTF, FactorizationSet, Factorization


def main(args):
    
    adata = sc.read("/home/dchafamo/final/data/combined_dataset_final_v2.h5ad")
    print(adata)


    sample_label = "donor"
    celltype_label = "cell_types_level_3"
    cell_types = ['CD4_T_cells', 'B_cells', 'CD8_T_cells', 'Macrophages', 'Fibroblasts',
           'Plasma_cells', 'mDC', 'Tumor', 'Monocytes', 'pDC', 'NK_cells', 'LEC',
           'BEC', 'FDC', 'T_other']

    sc_tensor = SingleCellTensor.from_anndata(adata,
        sample_label=sample_label,
        celltype_label=celltype_label,
        cell_types=cell_types,
        hgnc_approved_genes_only=True,
        normalize=True
    )
    sc_tensor.tensor = sc_tensor.tensor.round()
    
    if args.method == "btf":
        factorization_set = SingleCellBTF.factorize(
            sc_tensor = sc_tensor, 
            rank = [20], 
            model = 'gamma_poisson',
            n_restarts = args.num_restarts, 
            num_steps = args.num_steps,
            init_alpha = 1e2, 
            plot_var_explained = False
        )
    elif args.method == "hals":
        factorization_set = SingleCellBTF.factorize_hals(
            sc_tensor = sc_tensor, 
            num_steps = args.num_steps,
            rank = [16,24], 
            n_restarts = args.num_restarts, 
            sparsity_coefficients = [0.,0.,10.]
        )
        
    factorization_set.save(f"results/{args.output}_{args.method}_all_ctypes.pkl")


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument("-nr", "--num-restarts", default=50, type=int, help="number of restarts per rank")
    parser.add_argument("-ns", "--num-steps", default=1000, type=int, help="number of steps")
    parser.add_argument("--method", default="hals", type=str, help="use hals or btf to factorize")
    parser.add_argument("--output", default="factorization_set", type=str, help="name to save model by")
    args = parser.parse_args()
    assert args.method in ["hals", "btf"]
    main(args)