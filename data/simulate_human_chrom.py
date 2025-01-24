#%%
import msprime
import argparse
import demes
import numpy as np
import pandas as pd
import os
import subprocess
import stdpopsim
from tqdm import tqdm
#%%
# PATH_TO_RELATE = "/Users/afselim/Documents/RELATE/"
parser = argparse.ArgumentParser()
parser.add_argument("-input",type = str)
parser.add_argument("-output",type = str)
args = parser.parse_args()

input = args.input
output = args.output
# assert chrom_num > 0 and chrom_num < 23, "chromosome number should be between 1 and 22"

# %%
PATH_TO_RELATE = "/project2/jnovembre/afselim/relate/"
os.makedirs(output, exist_ok=True)
n=20
species = stdpopsim.get_species(id="HomSap")

graph = demes.load(input)
demography = msprime.Demography.from_demes(graph)
K=0
samples = {}
population_samples_dict = {}
for pop in demography.populations:
    if pop.default_sampling_time == 0:
        samples[pop.name] = n
        population_samples_dict[pop.name] = np.arange(2*n*K,2*n*(K+1))
        K+=1
n_samples = n*K
population_samples_tuples = [(pop, sample_id) for pop, sample_ids in population_samples_dict.items() for sample_id in sample_ids]
metadata_df = pd.DataFrame(population_samples_tuples, columns=["population", "sample_id"])
metadata_df.to_csv(os.path.join(output, "metadata.csv"), index=False)
N = []
mu_list = []
#%%
for chrom_num in tqdm(range(1,23)):
    print(f"simulating chromosome {chrom_num}")
    contig = species.get_contig(f"chr{chrom_num}","DeCodeSexAveraged_GRCh36") # starts from 1
    mu = contig.mutation_rate
    rate_map = contig.recombination_map
    sequence_length = contig.length
    ts = msprime.sim_ancestry(
        samples = samples,
        demography = demography,
        sequence_length=sequence_length,
        ploidy = 2,
        recombination_rate=rate_map
        )
    ts = msprime.sim_mutations(ts,rate=mu)

    N.append(ts.diversity(mode="branch")/2)
    mu_list.append(mu)
    rates = rate_map.rate
    positions = rate_map.position
    parsed_ratemap = "\n".join(["chrom pos rate"] + [f"chr_{chrom_num} {positions[i+1]} {rates[i]}" for i in range(len(rates))])
    with open(os.path.join(output,f"chrom_{chrom_num}.vcf"),"w") as f:
        ts.write_vcf(f)
    # rate_map = f"chr position COMBINED_rate(cM/Mb) Genetic_Map(cM)\nchr1 0 {r*10e6} 0\nchr1 {sequence_length} 0 {r*sequence_length*100}"
    with open(os.path.join(output,f"chrom_{chrom_num}.map"),"w") as f:
        f.write(parsed_ratemap)
    # msprime.RateMap.read_hapmap(os.path.join(output,"chrom.map"))

N = np.mean(N)
mu = np.mean(mu_list)
os.chdir(output)
for chrom_num in tqdm(range(1,23)):
    print(f"running RELATE for chromosome {chrom_num}")
    process_vcf = f'"{PATH_TO_RELATE}bin/RelateFileFormats" --mode ConvertFromVcf --haps chrom_{chrom_num}.hap --sample chrom_{chrom_num}.sample -i chrom_{chrom_num}'
    run_relate = f'"{PATH_TO_RELATE}bin/Relate" --mode All -N {N} -m {mu} --haps chrom_{chrom_num}.hap --sample chrom_{chrom_num}.sample --map chrom_{chrom_num}.map --output chrom_{chrom_num}'
    convert_to_treesequence = f'"{PATH_TO_RELATE}bin/RelateFileFormats" --mode ConvertToTreeSequence -i chrom_{chrom_num} -o chrom_{chrom_num}'

    cmd1 = subprocess.run(process_vcf, shell=True, capture_output=False)
    cmd2 = subprocess.run(run_relate, shell=True, capture_output=False)
    cmd3 = subprocess.run(convert_to_treesequence, shell=True, capture_output=False)
# %%
