#%%
import numpy as np
import pandas as pd
import os
import argparse
import glob
import tskit
from tqdm import tqdm
import random

def subset_metadata(metadata,populations = None,individuals_per_pop = None):
    if populations is None: # Select all populations
        populations = np.unique(metadata["population"])

    metadata_subset = metadata[metadata["population"].isin(populations)]
    if individuals_per_pop is not None:
        for population in populations:
            population_metadata = metadata_subset[metadata_subset['population']==population]
            n_samples_in_pop = len(population_metadata)
            if n_samples_in_pop < 2*individuals_per_pop: # If there are less than 2*individuals_per_pop samples in the population, select all samples
                continue 
            else:
                ind_ids = np.random.choice(np.arange(0, int(n_samples_in_pop/2)), size=individuals_per_pop, replace=False)
                haplotype_1_ids = 2*ind_ids
                haplotype_2_ids = 2*ind_ids+1
                subsample_ids = np.sort(np.concatenate([haplotype_1_ids,haplotype_2_ids]))
                subsample_ids = population_metadata.iloc[subsample_ids,:].sample_id.values
                drop_indices = population_metadata[~population_metadata["sample_id"].isin(subsample_ids)].index
                metadata_subset = metadata_subset.drop(drop_indices)
    metadata_subset = metadata_subset.reset_index(drop=True)
    return metadata_subset
#%%
argparser = argparse.ArgumentParser()
argparser.add_argument('-input', type=str, required=True)
argparser.add_argument('-output', type=str, required=True)
argparser.add_argument('-metadata', type=str)
argparser.add_argument('-populations', type=str, required=True)
argparser.add_argument('-individuals_per_pop', type=int, required=True)


args = argparser.parse_args()
input_path = args.input
output_path = args.output
metadata_path = args.metadata
populations = args.populations
individuals_per_pop = args.individuals_per_pop
#%%
if metadata_path is None:
    metadata_path = os.path.join(input_path,"metadata.csv")
print(metadata_path)
metadata = pd.read_csv(metadata_path)
if populations is None:
    populations = np.unique(metadata.population)
else:
    populations = populations.split(',')

os.makedirs(output_path,exist_ok=True)
#%%
print("subsampling from trees")
metadata_subset = subset_metadata(metadata,populations,individuals_per_pop)
ts_files = glob.glob(f"{input_path}/*trees")
for ts_file in tqdm(ts_files):
    ts = tskit.load(ts_file)
    ts = ts.simplify(metadata_subset.sample_id.values)
    ts.dump(os.path.join(output_path,os.path.basename(ts_file)))

metadata_subset.sample_id=metadata_subset.index
metadata_subset.to_csv(f"{output_path}/metadata.csv",index=False)
# %%
