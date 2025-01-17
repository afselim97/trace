#%%
import msprime
import argparse
import demes
import numpy as np
import os
import subprocess
#%%
parser = argparse.ArgumentParser()
parser.add_argument("-input",type = str)
parser.add_argument("-output",type = str)
args = parser.parse_args()

input = args.input
output = args.output

os.makedirs(output)
# %%
n=10
mu = 1.3e-8
r = 1.3e-8
sequence_length = 4e7

graph = demes.load(input)
demography = msprime.Demography.from_demes(graph)
K=0
samples = {}
population_samples_dict = {}
for pop in demography.populations:
    if pop.default_sampling_time == 0:
        samples[pop.name] = n
        population_samples_dict[pop.name] = np.arange(n*K,n*(K+1))
        K+=1
n_samples = n*K

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
PATH_TO_RELATE = "/Users/afselim/Documents/RELATE/"
os.chdir(output)
#%%
process_vcf = f'"{PATH_TO_RELATE}bin/RelateFileFormats" --mode ConvertFromVcf --haps chrom.hap --sample chrom.sample -i vcf_format'
run_relate = f'"{PATH_TO_RELATE}bin/Relate" --mode All -N {N} -m {mu} --haps chrom.hap --sample chrom.sample --map chrom.map --output relate_inference'
convert_to_treesequence = f'"{PATH_TO_RELATE}bin/RelateFileFormats" --mode ConvertToTreeSequence -i relate_inference -o ts_relate'

cmd1 = subprocess.run(process_vcf, shell=True, capture_output=True, text=True)
cmd2 = subprocess.run(run_relate, shell=True, capture_output=True, text=True)
cmd3 = subprocess.run(convert_to_treesequence, shell=True, capture_output=True, text=True)

# %%
