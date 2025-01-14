#!/usr/bin/env python3
#%%
import argparse
import numpy as np
import demes
import glob
import tskit
from typing import List
from tqdm import tqdm
from trace import *
try:
    import msprime
except:
    print("cannot simulate")

def calculate_L_list(num_trees:List[int],L:int) -> List[int]:
    """Calculates a list of the number of trees to sample from each tree sequence (representing a chromosome part)
        Draws proportionally larger numbers from larger chromosomes, ensuring total sums to L

    Args:
        num_trees (List[int]): Th enumber of trees in each tree sequence
        L (int): total number of trees desired

    Returns:
        List[int]: The number of trees to be extracted from each tree sequence
    """
    if L is not None:
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

def prepare_from_simulation(demes_file_path: str,n: int,L: int):
    """prepares an iterator of tree sequences given a demes file to simulate.

    Args:
        demes_file_path (str): The demography to be simulated in demes format
        n (int): number of haploid samples per population in given demography
        L (int): number of independent trees to simulate

    Returns:
        iterator: an iterator of tree sequences
        List[int]: a list of size L, where each entry is 2 (a single tree is taken from each simulation to ensure total independence)
        int: the total number of samples in the simulation
    """
    if "SS" in demes_file_path: ## simulate only 2 samples per deme for stepping stones models
        n=2
    try:
        graph = demes.load(demes_file_path)
    except:
        print("Invalid demes file")
        return
    demography = msprime.Demography.from_demes(graph)
    K=0
    samples = {}
    for pop in demography.populations:
        if pop.default_sampling_time == 0:
            samples[pop.name] = n
            K+=1
    n_samples = n*K

    ts_list = msprime.sim_ancestry(
        samples = samples,
        demography = demography,
        sequence_length=1,
        ploidy = 1,
        num_replicates = L
        )

    L_list = [1]*L
    
    return ts_list,L_list,n_samples

def prepare_from_inference(inferred_trees_dir: str,L: int):
    ts_files = glob.glob(f"{inferred_trees_dir}/*trees")
    num_trees = [tskit.load(ts_file).num_trees for ts_file in tqdm(ts_files)]
    n_samples = tskit.load(ts_files[0]).n_samples
    if L is None:
        L_list = (np.array(num_trees)/n_samples).astype(int)
    else:
        L_list = calculate_L_list(num_trees,L)

    ts_list = (tskit.load(ts_file) for ts_file in ts_files)
    return ts_list,L_list,n_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", type="str", required=True, help="path to demes file or to directory with inferred trees")
    parser.add_argument("-output", type="str", required=True, help="output path")
    parser.add_argument("-mode", type="str", default="inferred", help="either 'simulated' or 'inferred'.")
    parser.add_argument("-L", type=int, help="number of independent trees to sample or simulate")
    parser.add_argument("-max_time", type=float, help="maximum time to compute rates and embeddings")
    parser.add_argument("-num_timepoints", type=int, defualt=100, help="number of timepoints to use")
    parser.add_argument("-delta", type=float, help="the timw width of each window")
    parser.add_argument("-log_time", action="store_true", help="use this flag to have log-spaced time points")

    args = parser.parse_args()
    
    input = args.input
    output = args.output
    mode = args.mode
    L = args.L
    max_time = args.max_time
    num_timepoints = args.num_timepoints
    delta = args.delta
    log_time = args.log_time

    assert mode == "inferred" or mode == "simulated", "mode must be 'simulated' or 'inferred'"
    if mode == "simulated":
        if L is None:
            L=10000
        n=50
        ts_list,L_list,n_samples = prepare_from_simulation(demes_file_path =input, n=n, L=L)
    elif mode == "inferred":
        ts_list,L_list,n_samples = prepare_from_inference(inferred_trees_dir =input, L=L)

    trace_instance = trace(ts_list,L_list,n_samples)
    


if __name__ == "__main__":
    main()