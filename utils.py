import pandas as pd
import numpy as np

def get_cell_type_expression(cell_adata, cell_type_col, cell_types_to_exclude=[], mode='mean'):

    unique_cell_types = np.unique(cell_adata.obs[cell_type_col])
    type_exp_dict = {}
    for cell_type in unique_cell_types:
        if cell_type=='unassigned':
            continue
        if cell_type in cell_types_to_exclude:
            continue
        cell_indexes = cell_adata.obs[cell_adata.obs[cell_type_col]==cell_type].index.tolist()
        cell_indexes = [int(cell_index) for cell_index in cell_indexes]
        reduced_expression = cell_adata.X[cell_indexes]
        if mode=='mean':
            type_exp_dict[cell_type] = reduced_expression.mean(axis=0)
        elif mode=='median':
            type_exp_dict[cell_type] = np.median(reduced_expression, axis=0)
        
    expression_frame = pd.DataFrame(type_exp_dict).T

    return expression_frame