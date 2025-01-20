#%%
import msprime
import argparse
import demes
import numpy as np
import pandas as pd
import os
import subprocess
#%%
PATH_TO_RELATE = "/Users/afselim/Documents/RELATE/"
parser = argparse.ArgumentParser()
parser.add_argument("-input",type = str)
parser.add_argument("-output",type = str)
args = parser.parse_args()

input = args.input
output = args.output

os.makedirs(output, exist_ok=True)
# %%
n=10
mu = 1.3e-8
r = 1.3e-8
sequence_length = 1e8

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

print("simulating trees")
ts = msprime.sim_ancestry(
    samples = samples,
    demography = demography,
    sequence_length=sequence_length,
    ploidy = 2,
    recombination_rate=r
    )
ts = msprime.sim_mutations(ts,rate=mu)
N = ts.diversity(mode="branch")/2
#%%
with open(os.path.join(output,"vcf_format.vcf"),"w") as f:
    ts.write_vcf(f)
rate_map = f"chr position COMBINED_rate(cM/Mb) Genetic_Map(cM)\nchr1 0 {r*10e6} 0\nchr1 {sequence_length} 0 {r*sequence_length*100}"
with open(os.path.join(output,"chrom.map"),"w") as f:
    f.write(rate_map)
# msprime.RateMap.read_hapmap(os.path.join(output,"chrom.map"))

# %%
os.chdir(output)
#%%
process_vcf = f'"{PATH_TO_RELATE}bin/RelateFileFormats" --mode ConvertFromVcf --haps chrom.hap --sample chrom.sample -i vcf_format'
run_relate = f'"{PATH_TO_RELATE}bin/Relate" --mode All -N {N} -m {mu} --haps chrom.hap --sample chrom.sample --map chrom.map --output relate_inference'
convert_to_treesequence = f'"{PATH_TO_RELATE}bin/RelateFileFormats" --mode ConvertToTreeSequence -i relate_inference -o ts_relate'

cmd1 = subprocess.run(process_vcf, shell=True, capture_output=False)
cmd2 = subprocess.run(run_relate, shell=True, capture_output=False)
cmd3 = subprocess.run(convert_to_treesequence, shell=True, capture_output=False)

# %%
