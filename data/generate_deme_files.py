#%%
import msprime
import demes

demes_dir = "demes/"
#%%
def pop_split(N0,N1,N2,t_split,t_mig=0,mig_duration=0,m=0):
    """Creates a demography object of a population split that gave rise to 2 subpopulations, with no outgroup

    Args:
        N (int): effective population size of each subpopulation
        n (int): number of sampled individuals from each subpopulation
        t_split (float): time since the split
        t_mig (float): start time of a migration pulse
        mig_duration (float): duration of the migration pulse
        m (float): migration rate during the pulse

    Returns:
        tskit.demography: a demography object with the specified population history; 2 splits, a recent one, and a more distant one that defines an outgroup
        dict: number of samples individuals from each subpopulation 
    """
    demography = msprime.Demography()
    demography.add_population(name="main", initial_size= N0)

    demography.add_population(name="A", initial_size= N1, default_sampling_time = 0)
    demography.add_population(name="B", initial_size= N2, default_sampling_time = 0)

    demography.add_population_split(time=t_split, derived=["A", "B"], ancestral="main")

    if m!=0:
        assert t_mig!=0 and mig_duration!=0
        demography.add_symmetric_migration_rate_change(time=t_mig,populations=['A','B'],rate=m)
        demography.add_symmetric_migration_rate_change(time=t_mig+mig_duration,populations=['A','B'],rate=0)

    demography.sort_events()
    
    return demography

def pop_split_admix(N0,N1,N2,N_admix,t_split,t_admix,admix_prop=0.5):
    
    demography = msprime.Demography()
    demography.add_population(name="main", initial_size= N0)

    demography.add_population(name="A", initial_size= N1, default_sampling_time = 0)
    demography.add_population(name="B", initial_size= N2, default_sampling_time = 0)
    demography.add_population(name="C", initial_size= N_admix, default_sampling_time = 0)

    demography.add_population_split(time=t_split, derived=["A", "B"], ancestral="main")

    demography.add_admixture(time=t_admix,derived="C",ancestral=["A","B"],proportions=[admix_prop,1-admix_prop])

    demography.sort_events()
    
    return demography

def pop_split_2(N_dict,t_split_1,t_split_2):
    
    demography = msprime.Demography()
    demography.add_population(name="main", initial_size= N_dict["main"])

    demography.add_population(name="A", initial_size= N_dict["A"], default_sampling_time = 0)
    demography.add_population(name="B", initial_size= N_dict["B"], default_sampling_time = 0)
    demography.add_population(name="C", initial_size= N_dict["C"], default_sampling_time = 0)
    demography.add_population(name="temp_pop", initial_size= N_dict["temp_pop"])

    demography.add_population_split(time=t_split_1+t_split_2, derived=["temp_pop", "C"], ancestral="main")
    demography.add_population_split(time=t_split_1, derived=["A", "B"], ancestral="temp_pop")

    demography.sort_events()
    
    return demography

def recent_mig_change(N,t_split_1,t_split_2,t_change,m1,m2):
    demography = msprime.Demography()
    demography.add_population(name="main", initial_size= N)

    demography.add_population(name="A", initial_size= N, default_sampling_time = 0)
    demography.add_population(name="B", initial_size= N, default_sampling_time = 0)
    demography.add_population(name="C", initial_size= N, default_sampling_time = 0)
    demography.add_population(name="temp_pop", initial_size= N)

    demography.add_population_split(time=t_split_2, derived=["temp_pop", "C"], ancestral="main")
    demography.add_population_split(time=t_split_1, derived=["A", "B"], ancestral="temp_pop")

    demography.set_symmetric_migration_rate(populations= ["B","C"],rate= m1)
    demography.add_symmetric_migration_rate_change(time= t_change, populations= ["B","C"], rate= 0)
    demography.add_symmetric_migration_rate_change(time= t_change, populations= ["B","A"], rate= m2)

    demography.sort_events()

    return demography
#%%
Ne=1e4

# 1-split
N0=2*Ne
N1=1*Ne
N2=0.5*Ne
t_split = Ne
demography = pop_split(N0,N1,N2,t_split)
graph = demography.to_demes()
demes.dump(graph, demes_dir + "split.yaml")

# 2-split_mig
t_split = Ne
t_mig = Ne/2
mig_duration = Ne/2
m=1e-3
demography = pop_split(N0,N1,N2,t_split,t_mig,mig_duration,m)
graph = demography.to_demes()
demes.dump(graph, demes_dir + "split_mig.yaml")

# 3- Admix
N_admix = 1.5*Ne
t_split = Ne
t_admix = Ne/2
demography = pop_split_admix(N0,N1,N2,N_admix,t_split,t_admix,admix_prop=0.2)
graph = demography.to_demes()
demes.dump(graph, demes_dir + "admix.yaml")

# 4- pop_split_2
N_dict = {"A": 0.5*Ne,
          "B": Ne,
          "C": 2*Ne,
          "main": Ne,
          "temp_pop": Ne}
t_split_1 = Ne
t_split_2 = 2*Ne
t_admix = Ne/2
demography = pop_split_2(N_dict,t_split_1,t_split_2)
graph = demography.to_demes()
demes.dump(graph, demes_dir + "two_split.yaml")

# %%
