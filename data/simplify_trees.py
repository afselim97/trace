#%%
import numpy as np
import pandas as pd
import os
import argparse
import glob
import tskit

def subset_metadata(metadata,populations,individuals_per_pop):
    metadata_subset = metadata[metadata['population'].isin(populations)]
    metadata_subset = metadata_subset.groupby('population').sample(individuals_per_pop,replace="False")
    metadata_subset["sample_id"] = metadata_subset["sample_id"].apply(lambda x: x - 1 if x % 2 != 0 else x)
    duplicated_rows = metadata_subset.copy()
    duplicated_rows["sample_id"] = duplicated_rows["sample_id"] + 1
    metadata_subset = pd.concat([metadata_subset, duplicated_rows]).sort_index().reset_index(drop=True)
    return metadata_subset
#%%
argparser = argparse.ArgumentParser()
argparser.add_argument('--input', type=str, required=True)
argparser.add_argument('--output', type=str, required=True)
argparser.add_argument('--metadata', type=str)
argparser.add_argument('--populations', type=str, required=True)
argparser.add_argument('--individuals_per_pop', type=int, required=True)


args = argparser.parse_args()
input_path = args.input
output_path = args.output
metadata_path = args.metadata
populations = args.populations
individuals_per_pop = args.individuals_per_pop

if metadata_path is None:
    metadata_path = os.path.join(input_path,"metadata.csv")
populations = populations.split(',')
metadata = pd.read_csv(metadata_path)
os.makedirs(output_path,exist_ok=True)
# %%
metadata_subset = subset_metadata(metadata,populations,individuals_per_pop)
ts_files = glob.glob(f"{input_path}/*trees")
for ts_file in ts_files:
    ts = tskit.load(ts_file)
    ts = ts.simplify(metadata_subset.sample_id.values)
    ts.dump(os.path.join(output_path,os.path.basename(ts_file)))

metadata_subset.sample_id=metadata_subset.index
metadata_subset.to_csv(f"{output_path}/metadata.csv",index=False)