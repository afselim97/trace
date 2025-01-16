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

def get_samples_dict(metadata_csv: pd.DataFrame) -> Dict[str,List[int]]:
    """Obtains a dictionary of sample ids in populations given an extrenal csv of the data. Keys are population labels and values are sample_ids.

    Args:
        metadata_csv (pd.DataFrame): input csv must have columns with names "sample_id"

    Returns:
        Dict: a dictionary with population names as keys and list of sample ids as values
    """
    assert np.all(np.in1d(np.array(list(metadata_csv.columns)),np.array(["sample_id","population"])))
    populations = np.unique(metadata_csv.population.values)
    samples_dict = {}
    for population in populations:
        sample_ids = metadata_csv[metadata_csv.population == population].sample_id.values
        samples_dict[population] = sample_ids
    return samples_dict
