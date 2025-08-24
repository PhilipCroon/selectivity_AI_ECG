"""
Heatmaps, specifically for the sensiticity anlysis
"""

# %%
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


# 1. Settings ---------------------------------------------------------------

base_path = "/home/pmc57/PheWas_AI_ECG/tabs"

data_dict_yale = {
    "Sex": "SENS{}yale_results_preds_image_MaleSex_YNHH_min20.csv",
    "LVSD": "SENS{}yale_results_preds_image_Under40_YNHH_min20.csv",
    "AS":  "SENS{}yale_results_preds_image_ModerateOrSevereAS_YNHH_min20.csv",
    "MR":  "SENS{}yale_results_preds_image_ModerateOrSevereMR_YNHH_min20.csv",
    "LVH": "SENS{}yale_results_preds_image_HCM_LVDD_IVSd15_IntermediateAsFalse_YNHH_min20.csv",
    "SHD": "SENS{}yale_results_preds_image_cmp_HCM_LVDD_ModtoSevVD_EF40_YNHH_min20.csv",
}


exp_codes  = [4,3,2,1]
exp_labels = {4: "All", 3: "Max 3", 2: "Max 2", 1: "Max 1"}

# Curated phenotype descriptions
sig_list_curated_sub = [
    "Heart failure",
    "Aortic valve disease",
    "Mitral valve disease",
    "Left ventricular hypertrophy",
]

# Branding colors
color_ynhh = (0.050, 0.164, 0.376)  # Yale Deep Navy Blue
color_comm = (0.565, 0.741, 0.565)  # Soft Sage Green

# 2. Load & combine all experiments -----------------------------------------

records = []
for code in exp_codes:
    label = exp_labels[code]
    for model_name, pattern in data_dict_yale.items():
        fname = os.path.join(base_path, pattern.format(code))
        df = pd.read_csv(fname)
        df['description'] = df['description'].map({'Heart failure NOS': 'Heart failure',
                                                   'Aortic valve disease': "Aortic valve disease",
                                                          'Cardiomegaly': "Left ventricular hypertrophy",
                                                          }).fillna(df['description'])
        df['Experiment'] = label
        df['Model']      = model_name
        records.append(df[["Model", "description", "OR", "Experiment"]])

all_df = pd.concat(records, ignore_index=True)

# 3. Filter to curated phenotypes -------------------------------------------

filtered = all_df[all_df['description'].isin(sig_list_curated_sub)].copy()

# 4. Compute global vmin/vmax across all ORs --------------------------------

all_vals = filtered['OR'].dropna().values
vmin, vmax = all_vals.min(), all_vals.max()

# %% 5. Plot 2×2 heatmaps -------------------------------------------------------

ordered_models   = list(data_dict_yale.keys())
ordered_outcomes = sig_list_curated_sub

# Diverging colormap (blue-white-green)
custom_cmap = LinearSegmentedColormap.from_list(
    'diverging', [color_ynhh, (1,1,1), color_comm], N=256
)

fig, axes = plt.subplots(2, 2, figsize=(12, 10),
                         gridspec_kw={'hspace':0.5, 'wspace':-0.5})
axes = axes.flatten()

for ax, code in zip(axes, exp_codes):
    label = exp_labels[code]
    df_sub = filtered[filtered['Experiment'] == label]
    
    # pivot so rows=Model, cols=description
    mat = df_sub.pivot_table(
        index='Model', columns='description', values='OR', aggfunc='mean'
    ).reindex(index=ordered_models, columns=ordered_outcomes)
    
    # fill missing with OR=1 for neutrality
    mat = mat.fillna(1.0)
    
    sns.heatmap(
        mat,
        ax=ax,
        cmap=custom_cmap,
        center=1.0,
        vmin=vmin, vmax=vmax,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor='gray',
        cbar=False,
        square=True,
        annot_kws={"size": 7}
    )
    ax.set_title(f"Experiment {code} ({label})", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.tick_params(axis='x', labelsize=8, length=2)
    
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.tick_params(axis='y', labelsize=8, length=2)

for ax in axes.reshape(-1, 2)[:,0]:   # picks the first column of subplots
    ax.set_ylabel("Models", fontsize=9)
    
for ax in axes[2:]:
    ax.set_xlabel("Phenotypes", fontsize=9)

# Adjust layout
# plt.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.)
plt.subplots_adjust(left=0.1,
                    right=0.85,
                    top=0.95,
                    bottom=0.2,   # increase this from e.g. 0.1 to 0.2
                    hspace=0.3,
                    wspace=0.1)

# Save & display
output_file = os.path.join(base_path, "phewash_sensitivity_heatmaps.pdf")
plt.savefig(output_file, dpi=300)
plt.show()
# %%

# %% heatmap + HCM


data_dict_yale = {
    "AS": "yale_results_preds_image_ModerateOrSevereAS_YNHH_min20.csv",
    "LVH": "yale_results_preds_image_HCM_LVDD_IVSd15_IntermediateAsFalse_YNHH_min20.csv",
    "HCM": "yale_results_preds_image_HCM_NatCVR_YNHH_min20.csv",
}

sig_list_curated_sub = [
    "AS",
    "LVH",
    "HOCM",
    "HCM"
]
# Branding colors
color_ynhh = (0.050, 0.164, 0.376)  # Yale Deep Navy Blue
color_comm = (0.565, 0.741, 0.565)  # Soft Sage Green

# === 2. Load and combine all model results ===
records = []
for model_name, fname in data_dict_yale.items():
    path = os.path.join(base_path, fname)
    try:
        df = pd.read_csv(path)
        df['description'] = df['description'].map({'Heart failure NOS': 'HF',
                                                   'Aortic valve disease': "AS",
                                                          'Cardiomegaly': "LVH",
                                                          "Other hypertrophic cardiomyopathy": "HCM",
                                                          'Hypertrophic obstructive cardiomyopathy': "HOCM"
                                                          }).fillna(df['description'])
        df['Model'] = model_name
        records.append(df[["Model", "description", "OR"]])
    except Exception as e:
        print(f"⚠️ Could not load {model_name}: {e}")


all_df = pd.concat(records, ignore_index=True)

# === 3. Filter to selected phenotypes ===
filtered = all_df[all_df['description'].isin(sig_list_curated_sub)].copy()

print(filtered)

# === 4. Compute color scaling ===
all_vals = filtered['OR'].dropna().values
vmin = 1.0                     # Force scale to start at 1 (neutral)
vmax = np.percentile(all_vals, 99) 

# === 5. Plot heatmap ===
ordered_models = list(data_dict_yale.keys())
ordered_outcomes = sig_list_curated_sub

custom_cmap = LinearSegmentedColormap.from_list(
    'diverging', [color_comm, (1, 1, 1), color_ynhh], N=256
)

fig, ax = plt.subplots(figsize=(8, 6))
mat = filtered.pivot_table(
    index='Model', columns='description', values='OR', aggfunc='mean'
).reindex(index=ordered_models, columns=ordered_outcomes).fillna(1.0)

sns.heatmap(
    mat,
    ax=ax,
    cmap=custom_cmap,
    center=1.0,
    vmin=vmin,
    vmax=vmax,
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    linecolor='gray',
    cbar=True,
    cbar_kws={"shrink": 0.5, "label": "Odds Ratio (OR)", "pad": 0.02},
    square=True,
    annot_kws={"size": 8}
)

ax.set_title("Heatmap including HCM – YNHH", fontsize=11, fontweight='bold')
ax.set_xlabel("Phenotypes", fontsize=9)
ax.set_ylabel("Models", fontsize=9)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
ax.tick_params(axis='both', labelsize=8, length=2)

plt.tight_layout()
output_file = os.path.join(base_path, "extended_heatmap_LVH.pdf")
plt.savefig(output_file, dpi=300)
plt.show()