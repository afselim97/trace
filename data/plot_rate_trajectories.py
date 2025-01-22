#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler
from matplotlib.cm import get_cmap
def set_custom_rcparams():

    # -- Axes --
    rcParams['axes.spines.bottom'] = True
    rcParams['axes.spines.left'] = True
    rcParams['axes.spines.right'] = False
    rcParams['axes.spines.top'] = False
    rcParams['axes.grid'] = False
    rcParams['axes.grid.axis'] = 'y'
    rcParams['grid.color'] = 'black'
    rcParams['grid.linewidth'] = 0.5
    rcParams['axes.axisbelow'] = True
    rcParams['axes.linewidth'] = 1
    rcParams['axes.ymargin'] = 0
    # -- Ticks and tick labels --
    rcParams['axes.edgecolor'] = 'black'
    rcParams['xtick.color'] = 'black'
    rcParams['ytick.color'] = 'black'
    rcParams['xtick.major.width'] = 2
    rcParams['ytick.major.width'] = 0
    rcParams['xtick.major.size'] = 5
    rcParams['ytick.major.size'] = 0
    # -- Fonts --
    rcParams['font.size'] = 12
    rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
    rcParams['text.color'] = 'black'
    rcParams['axes.labelcolor'] = 'black'
    # -- Figure size --
    rcParams['figure.figsize'] = (6, 6)
    # -- Saving Options --
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.dpi'] = 500
    rcParams['savefig.transparent'] = False
    # -- Plot Styles --
    rcParams['lines.linewidth'] = 3
    navy = (56 / 256, 74 / 256, 143 / 256)
    teal = (106 / 256, 197 / 256, 179 / 256)
    pink = [199 / 255, 99 / 255, 150 / 255]
    orange = (255 / 256, 153 / 256, 102 / 256)
    purple = (153 / 256, 102 / 256, 204 / 256)
    green = (102 / 256, 204 / 256, 153 / 256)
    yellow = (255 / 256, 204 / 256, 102 / 256)
    red = (204 / 256, 102 / 256, 102 / 256)
    rcParams['axes.prop_cycle'] = cycler(color=[teal, pink, navy, orange, purple, green, yellow, red])
     
    # wohns dataset
    # continuous_palette=plt.colormaps.get_cmap('viridis').resampled(216).colors 
    # rcParams['axes.prop_cycle'] = cycler(color=continuous_palette)
set_custom_rcparams()
# %%
result_dir = "results/sim/"
df = pd.read_parquet(result_dir+ "data.parquet")
df = df[df['sample_1_inx'] < df['sample_2_inx']]
df['population_combination'] = df['population_1'].astype(str) + "_" + df['population_2'].astype(str)

# %%
unique_pop_combs = np.unique(df['population_combination'])
cmap = get_cmap("rainbow")
palette = [cmap(i / len(unique_pop_combs)) for i in range(len(unique_pop_combs))]
color_dict = dict(zip(unique_pop_combs, palette))

sample_1_inxs = np.unique(df["sample_1_inx"])
sample_2_inxs = np.unique(df["sample_2_inx"])
fig,ax = plt.subplots(1,1,figsize=(10,10))
for sample_1_inx in sample_1_inxs:
    for sample_2_inx in sample_2_inxs:
            df_sub = df[(df["sample_1_inx"] == sample_1_inx) & (df["sample_2_inx"] == sample_2_inx)]
            if df_sub.shape[0] > 0:
                ax.plot(df_sub["time"],df_sub["raw_rate"],lw=0.2,c = color_dict[df_sub["population_combination"].values[0]])

handles = [plt.Line2D([0], [0], color=color_dict[pop], lw=2) for pop in unique_pop_combs]
labels = unique_pop_combs
ax.legend(handles, labels, title='Population Combination', loc='upper right')
ax.set_ylim(0,0.00015)
plt.title('Raw Rate Trajectories')
plt.xlabel('Time')
plt.ylabel('Raw Rate')
plt.show()
# %%
