#%%
import numpy as np
import polars as pl
from typing import List,Dict
import json
import tskit
import pandas as pd
from copy import copy
import random
from sklearn.manifold import smacof
from tqdm import tqdm
from numpy.typing import NDArray

#%%
# Functions to convert between a matrix and a dataframe
def matrix_to_df(matrix: np.ndarray) -> pl.DataFrame:
    # Processes up to a 4d matrix into a dataframe of: [row,column,depth,element_1,...,.element_p] when the last dimension has p components
    d = matrix.ndim
    assert d<=4
    extra_dim = 4-d
    new_shape = (...,) + (np.newaxis,) * extra_dim
    matrix = matrix[new_shape]
    r,c,d,p = matrix.shape
    columns = ['row', 'column', 'depth'] + [f'element_{i+1}' for i in range(p)]
    
    #Creating indices
    row_idx, col_idx, depth_idx = np.meshgrid(np.arange(r), np.arange(c), np.arange(d), indexing='ij')
    row_idx_flat = row_idx.flatten()
    col_idx_flat = col_idx.flatten()
    depth_idx_flat = depth_idx.flatten()

    #Reshaping the matrix into a 2d matrix
    elements_flat = matrix.reshape(-1, p)
    data = np.column_stack((row_idx_flat, col_idx_flat, depth_idx_flat, elements_flat))
    df = pl.DataFrame(data, schema=columns)

    df = df.with_columns(
        pl.col("row").cast(pl.Int32),
        pl.col("column").cast(pl.Int32),
        pl.col("depth").cast(pl.Int32)
    )

    return df

def df_to_matrix(df: pl.DataFrame) -> np.ndarray:
    r = df["row"].max() + 1
    c = df["column"].max() + 1
    d = df["depth"].max() + 1

    element_cols = [col for col in df.columns if col.startswith("element")]
    p = len(element_cols)
    elements = df[element_cols].to_numpy()

    matrix = elements.reshape(r,c,d,p)

    return matrix

def create_data_df(matrix: np.ndarray, times: np.ndarray,delta: float, samples_dict: dict) -> pl.DataFrame:
    # Convert matrix to Polars DataFrame
    df = matrix_to_df(matrix)

    # Rename columns
    df = df.rename({
        "row": "sample_1_inx",
        "column": "sample_2_inx",
        "depth": "time_inx",
        "element_1": "num_uncoalesced",
        "element_2": "num_coal_events",
        "element_3": "raw_rate"
    })

    # Calculate delta (difference between times)
    # delta = times[1:] - times[:-1]

    # Create dictionaries for mappings
    sample_to_pop_dict = {sample: pop for pop, samples in samples_dict.items() for sample in samples}
    # times_dict = {i: t for i, t in enumerate(times[:-1])}
    times_dict = {i: t for i, t in enumerate(times)}
    # delta_dict = {i: d for i, d in enumerate(delta)}

    # Map values to new columns
    df = df.with_columns([
        pl.col("sample_1_inx").map_elements(lambda x: sample_to_pop_dict.get(x, "unknown"),return_dtype=str).alias("population_1"),
        pl.col("sample_2_inx").map_elements(lambda x: sample_to_pop_dict.get(x, "unknown"),return_dtype=str).alias("population_2"),
        pl.col("time_inx").map_elements(lambda x: times_dict.get(x, np.nan),return_dtype=float).alias("time")
        # pl.col("time_inx").map_elements(lambda x: delta_dict.get(x, np.nan),return_dtype=float).alias("delta")  # Using np.nan for missing values
    ])
    # Perform calculation and add new column
    df = df.with_columns([
        (np.sqrt((pl.col("num_coal_events") + 1) / (delta * (pl.col("num_uncoalesced") + 1) ** 2))).alias("raw_rate_std")
    ])

    return df


def calculate_L_list(num_trees:List[int],L:int) -> List[int]:
    """Calculates a list of the number of trees to sample from each tree sequence (representing a chromosome part)
        Draws proportionally larger numbers from larger chromosomes, ensuring total sums to L

    Args:
        num_trees (List[int]): Th enumber of trees in each tree sequence
        L (int): total number of trees desired

    Returns:
        List[int]: The number of trees to be extracted from each tree sequence
    """
    proportions = np.array(num_trees) / np.sum(num_trees)
    raw_allocation = proportions * L
    initial_allocation = np.floor(raw_allocation).astype(int)
    remaining = L - np.sum(initial_allocation)
    fractions = raw_allocation - np.floor(raw_allocation)
    indices_sorted_by_fraction = np.argsort(fractions)[::-1]
    for i in range(remaining):
        initial_allocation[indices_sorted_by_fraction[i]] += 1

    final_allocation = initial_allocation
    return final_allocation

def get_samples_ts(annotated_ts: tskit.TreeSequence) -> Dict[str,List[int]]:
    """parses the metadata of a tree sequence to obtain a dictionary of sample ids belonging to each population

    Args:
        annotated_ts (tskit.TreeSequence): A tree sequence with population names in the metadata

    Returns:
        Dict: a dictionary with population names as keys and list of sample ids as values
    """
    samples_dict={}
    for pop in annotated_ts.populations():
        metadata = pop.metadata
        if isinstance(metadata,bytes):
            metadata = json.loads(metadata.decode('utf-8'))
        pop_name = metadata["name"]
        if len(annotated_ts.samples(pop.id)) > 0:
            if pop_name in samples_dict.keys():
                samples_dict[pop_name] = np.concatenate((samples_dict[pop_name],annotated_ts.samples(pop.id)))
            else:
                samples_dict[pop_name] = annotated_ts.samples(pop.id)
    return samples_dict

def get_samples_csv(metadata_csv: pd.DataFrame) -> Dict[str,List[int]]:
    """Obtains a dictionary of sample ids in populations given an extrenal csv of the data (in the case where teh tree sequence is not annotated)
        This is the case where each individual has a single id rather than each haplotype

    Args:
        metadata_csv (pd.DataFrame): input csv must have columns with names "sample_id" (which is the id of an individual that has 2 haplotypes) and "population"

    Returns:
        Dict: a dictionary with population names as keys and list of sample ids as values
    """
    assert np.all(np.in1d(np.array(list(metadata_csv.columns)),np.array(["sample_id","population"])))
    populations = np.unique(metadata_csv.population.values)
    samples_dict = {}
    for population in populations:
        sample_ids = metadata_csv[metadata_csv.population == population].sample_id.values
        # haplotype1 = 2*individual_ids
        # haplotype2 = 2*individual_ids+1
        # sample_ids = np.concatenate(haplotype1,haplotype2)
        samples_dict[population] = sample_ids
    return samples_dict

def rates_to_distance(rates_over_time):
    n = rates_over_time.shape[0]
    rates_over_time[rates_over_time==0] = np.min(rates_over_time[rates_over_time!=0])*0.5
    D = -np.log(rates_over_time/np.max(rates_over_time))
    D[np.arange(n),np.arange(n),:] = 0

    return D

def compute_embeddings_over_time(rates_over_time,n_components: int = 2):
    """Computes PCs over time given a matrix over time

    Args:
        rates_over_time (np.ndarray): different matrices over the third dimension is over different time points
        n_components (int, optional): num components to compute PCA. Defaults to 2.
    """

    n = rates_over_time.shape[0]
    n_timepoints = rates_over_time.shape[-1]
    X_over_time = np.zeros((n,n_components,n_timepoints))
    D = rates_to_distance(rates_over_time)

    for k in tqdm(range(n_timepoints)):
        inx = n_timepoints-k-1
        if k==0:
            X, _ = smacof(D[:,:,inx], n_components=n_components)
        else:
            X, _ = smacof(D[:,:,inx], n_components=n_components,init=init)
        X_over_time[:,:,inx] = X
        init = X
    return X_over_time

def create_embeddings_df(matrix: np.ndarray, times: np.ndarray, samples_dict: dict) -> pl.DataFrame:
    # Convert matrix to Polars DataFrame
    df = matrix_to_df(matrix)

    # Rename columns
    df = df.rename({
        "row": "sample_id",
        "column": "component",
        "depth": "time_inx",
        "element_1": "embedding"
    })

    # Calculate delta (difference between times)

    # Create dictionaries for mappings
    sample_to_pop_dict = {sample: pop for pop, samples in samples_dict.items() for sample in samples}
    times_dict = {i: t for i, t in enumerate(times)}

    # Map values to new columns
    df = df.with_columns([
        pl.col("sample_id").map_elements(lambda x: sample_to_pop_dict.get(x, "unknown"),return_dtype=str).alias("population"),
        pl.col("time_inx").map_elements(lambda x: times_dict.get(x, np.nan),return_dtype=float).alias("time"),
    ])
    return df

import numpy as np

def procrustes_adjustment(X, Y):
    
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    
    
    cov_matrix = np.dot(Y_centered.T, X_centered)
    
    
    U, _, Vt = np.linalg.svd(cov_matrix)
    rotation_matrix = np.dot(U, Vt)
    
    
    Y_rotated = np.dot(Y_centered, rotation_matrix)
    
    return Y_rotated

