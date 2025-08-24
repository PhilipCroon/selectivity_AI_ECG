"""
Figures for the experimental models
"""

# %%
import pandas as pd
import numpy as np
import os
import plot_misc
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib as mpl
import plot_misc.heatmap as heatmap
import plot_misc.barchart as barchart
import plot_misc.utils.utils as pm_utils
from matplotlib.ticker import MaxNLocator
from typing import List, Optional, Any
import project_constants as project
from scipy.stats import fisher_exact
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib

# %%
# Define base path
base_path = "/home/pmc57/PheWas_AI_ECG/tabs"

# Define the cohort names
cohorts = ["YNHH", "Community", "NEMG"]


cohort_map = {
    "YNHH": " \nYale New Haven\nHospitals",
    "UKB": "UK Bio\n ",
    "NEMG": "Outpatient clinics",
    "Community": "Community hospitals"
}

cohort_map_box = {
    "YNHH": " Yale New Haven Hospitals",
    "UKB": "UK Biobank",
    "NEMG": "Outpatient clinics",
    "Community": "Community hospitals"
}  


data_dict_UKB_exp = {

    "Even month": "results_preds_image_EvenMonth_min20.csv",
    "Viral respiratory infection": "results_preds_Viral_respiratory_infection_min20.csv",
    "Transport accident": "results_preds_Transport_accident_min20.csv",
    "Headache": "results_preds_Headache_min20.csv",
    "Fracture of the lower leg": "results_preds_Fractures_min20.csv",
    "Bitten by dog": "results_preds_Bitten_by_dog_min20.csv",
    "Dermatophytosis": "results_preds_Dermatophytosis_min20.csv",
}

data_dict_yale = {
        "Even month": "yale_results_preds_image_EvenMonth_{}_min20.csv",
        "Viral respiratory infection": "yale_results_preds_Viral_respiratory_infection_{}_min20.csv",
         "Transport accident":'yale_results_preds_Transport_accident_{}_min20.csv',
         "Headache":'yale_results_preds_Headache_{}_min20.csv',
         "Fracture of the lower leg": 'yale_results_preds_Fractures_{}_min20.csv',
         "Bitten by dog": "yale_results_preds_Bitten_by_dog_{}_min20.csv",
        "Dermatophytosis": "yale_results_preds_Dermatophytosis_{}_min20.csv",
     }

# %%
color_ukb = (0.902, 0.298, 0.235)   # Warm Coral Red
color_ynhh = (0.050, 0.164, 0.376)  # Deep Navy Blue (Yale)
color_comm = (0.565, 0.741, 0.565)  # Soft Sage Green
color_nemg = (0.545, 0.247, 0.537)  # Warmer Eggplant Purple

cohort_colors = {
    "UKB": color_ukb,
    "YNHH": color_ynhh,
    "Community": color_comm,
    "NEMG": color_nemg
}

# Define group colors for pie charts
group_colors = {
    'circulatory system': (0.984, 0.502, 0.447),
    'endocrine/metabolic': (0.553, 0.827, 0.780),
    'mental disorders': (1.0, 1.0, 0.702),
    'congenital anomalies': (0.745, 0.729, 0.855),
    'genitourinary': (0.502, 0.694, 0.827),
    'sense organs': (0.992, 0.706, 0.384),
    'injuries & poisonings': (1.0, 0.6, 0.6), 
    'hematopoietic': (0.988, 0.804, 0.898),
    'respiratory': (0.851, 0.851, 0.851),
    'digestive': (0.737, 0.502, 0.741),
    'musculoskeletal': (0.8, 0.922, 0.773),
    'dermatologic': (1.0, 0.929, 0.435),
    'neurological': (0.580, 0.404, 0.741),
    'infectious diseases': (0.549, 0.337, 0.294),
    'neoplasms': (0.773, 0.694, 0.835),
    'symptoms': (0.894, 0.769, 0.580),
    'pregnancy complications': (0.702, 0.871, 0.412),
    True: "lightblue",
    False: "lightgrey"
}

# Define a fixed group order based on all possible categories across datasets
group_order = ['circulatory system', 'congenital anomalies', 
               'dermatologic', 'digestive', 'endocrine/metabolic', 'genitourinary', 
               'hematopoietic', 'infectious diseases', 'injuries & poisonings', 'mental disorders', 
               'musculoskeletal', 'neoplasms', 'neurological', 'pregnancy complications', 'respiratory', 
               'sense organs', 'symptoms']

# Ensure color mapping follows this order
group_colors_ordered = {group: group_colors.get(group, 'lightgrey') for group in group_order}

sig_dict = {}
data_total = []

# %%% Loop through cohorts and make piechart
for cohort in cohorts:
    all_data_list = []
    data_list = []
    data_list2 = []
    all_sig_list = []
    
    # Load the cohort-specific files
    for key, filename_template in data_dict_yale.items():
        file_path = os.path.join(base_path, filename_template.format(cohort))
        
        if os.path.exists(file_path):
            print(key)
            data = pd.read_csv(file_path)
            data['cohort'] = cohort
            # Compute Bonferroni threshold
            sig_threshold = 0.05 / len(data['description'].unique())
            print("sig", cohort,len(data['description'].unique()))
            print(f'{cohort}: n phetypes', len(data['description'].unique()))
            sig_dict[cohort] = sig_threshold
            
            # Store data
            all_data_list.append(data)
            data['snp'] = key
            data['coef'] = np.log(data['OR'])  # Log transformation
            
            # Filter significant values
            data2 = data[data['p'] < sig_threshold]
            all_sig_list.extend(data2['description'].to_list())
            
            data_list.append(data)
            data_list2.append(data2)
            data_total.append(data)
        else:
            print(f"Warning: File {file_path} not found.")

    # %%
  
  
#%% Piecharts UKB

# Load the cohort-specific files
all_data_list = []
data_list = []
data_list2 = []
all_sig_list = []
for key, filename_template in data_dict_UKB_exp.items():
    file_path = os.path.join(base_path, filename_template)
    
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        data['cohort'] = 'UKB'
        # Compute Bonferroni threshold
        sig_threshold = 0.05 / len(data['description'].unique())
        sig_dict['UKB'] = sig_threshold
        print(len(data['description'].unique()))
        print(f'UKB: n phetypes', len(data['description'].unique()))
        
        # Store data
        all_data_list.append(data)
        data['snp'] = key
        data['coef'] = np.log(data['OR'])  # Log transformation
        
        # Filter significant values
        data2 = data[data['p'] < sig_threshold]
        all_sig_list.extend(data2['description'].to_list())
        
        data_list.append(data)
        data_list2.append(data2)
        data_total.append(data)
    else:
        print(f"Warning: File {file_path} not found.")
        
merged_data_total = pd.concat(data_total) if data_list else pd.DataFrame()
# %% Volcano

def insert_newline_after_third_space(text):
    # Split the string by spaces
    words = text.split()
    
    # If the string has more than 3 words, insert a newline after the third word
    if len(words) > 3:
        # Join the first 3 words, add a newline, and then join the rest of the words
        return ' '.join(words[:3]) + '\n' + ' '.join(words[3:])
    else:
        # If there are 3 or fewer words, return the text unchanged
        return text

    

# **Define settings**
marker_size = 8
XLIM = [-3, 3]

# **Define cohort pairs for the single figure**
cohort_pair_sets = [
    ("YNHH", "Community"),  # Left column
    ("NEMG", "UKB")         # Right column
]
# %%
# **Get unique models from dataset**
unique_models = merged_data_total["snp"].unique()

# **Create the figure**
fig = plt.figure(figsize=(8, len(unique_models) * 4))  # Dynamic figure size
gs = gridspec.GridSpec(len(unique_models), 2, figure=fig, hspace=0.3, wspace=0.8)

# **Loop over each model to create individual plots in the grid**
print(unique_models)
for i, model in enumerate(unique_models):
    for j, (cohort_top, cohort_bottom) in enumerate(cohort_pair_sets):  # j=0 → Left, j=1 → Right
        print(cohort_top, cohort_bottom, model)
        ax = fig.add_subplot(gs[i, j])
        ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
        stop = sig_dict[cohort_top]
        sbottom = sig_dict[cohort_bottom]

        # **Filter data for both cohorts**
        data_top = merged_data_total[
            (merged_data_total["cohort"] == cohort_top) & (merged_data_total["snp"] == model)
        ].copy()

        data_bottom = merged_data_total[
            (merged_data_total["cohort"] == cohort_bottom) & (merged_data_total["snp"] == model)
        ].copy()

        # Make offset so won't fail in case of 0
        data_top["p"] += 1e-300
        data_bottom["p"] += 1e-300
        
        
        # Transform values
        
        # **Ensure p-values & OR are positive**
        data_top = data_top[(data_top["p"] > 0) & (data_top["OR"] > 0)]
        data_bottom = data_bottom[(data_bottom["p"] > 0) & (data_bottom["OR"] > 0)]

        # Step 3: Apply -log10 transformation after correction
        data_top["log_p"] = -np.log10(data_top["p"])
        data_bottom["log_p"] = -np.log10(data_bottom["p"])
        data_bottom["log_p"] = -data_bottom["log_p"]  # Flip for mirroring
        
        data_top["log_OR"] = np.log(data_top["OR"])
        data_bottom["log_OR"] = np.log(data_bottom["OR"])
        print(data_bottom)

        # **Determine y-axis limits**
        max_abs_y = max(data_top["log_p"].max(), abs(data_bottom["log_p"].min()))
        ax.set_ylim([-50, 50])
        ax.set_xlim(XLIM)

        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)

        # **Find significant points**
        sig_top = data_top["p"] < stop
        sig_bottom = data_bottom["p"] < sbottom

        # **Plot non-significant points (Gray)**
        ax.scatter(data_top["log_OR"], data_top["log_p"], color="gray", alpha=0.3, s=marker_size)
        ax.scatter(data_bottom["log_OR"], data_bottom["log_p"], color="gray", alpha=0.3, s=marker_size)

        # **Plot significant points**
        ax.scatter(data_top.loc[sig_top, "log_OR"], data_top.loc[sig_top, "log_p"], 
                   color=cohort_colors[cohort_top], alpha=0.7, label=cohort_top, s=marker_size)

        ax.scatter(data_bottom.loc[sig_bottom, "log_OR"], data_bottom.loc[sig_bottom, "log_p"], 
                   color=cohort_colors[cohort_bottom], alpha=0.7, label=cohort_bottom, s=marker_size)

        # Compute absolute log_OR distance from 0
        data_top["log_OR_distance"] = abs(data_top["log_OR"])
        data_bottom["log_OR_distance"] = abs(data_bottom["log_OR"])

        # **Sort first by p-value (ascending, i.e., most significant), then by largest |log_OR|**
        top_6_TOP = data_top.loc[sig_top].sort_values(by=["p", "log_OR_distance"], ascending=[True, False]).head(6)
        top_6_BOTTOM = data_bottom.loc[sig_bottom].sort_values(by=["p", "log_OR_distance"], ascending=[True, False]).head(6)

        # **Set spacing for labels**
        ymin, ymax = ax.get_ylim()
        y_positions_top = np.linspace(ymax * 0.9, ymax * 0.1, len(top_6_TOP))  # Space out labels better
        y_positions_bottom = np.linspace(ymin * 0.9, ymin * 0.1, len(top_6_BOTTOM))

        constant_x = ax.get_xlim()[1] * 1.05  # Place labels outside the plot

        annotations = []

        # **Top Cohort Labels**
        for (row, label_y) in zip(top_6_TOP.iterrows(), y_positions_top):
            x, y = row[1]['log_OR'], row[1]['log_p']
            formatted_label = insert_newline_after_third_space(row[1]['description'])
            ann = ax.annotate(
                formatted_label, 
                xy=(x, y),  
                xytext=(constant_x, label_y),  
                horizontalalignment="left", fontsize=5.5,
                arrowprops=dict(arrowstyle="-", color='black', linewidth=0.5, 
                                connectionstyle="angle,angleA=0,angleB=90,rad=0")  
            )
            annotations.append(ann)

        # **Bottom Cohort Labels**
        for (row, label_y) in zip(top_6_BOTTOM.iterrows(), y_positions_bottom):
            x, y = row[1]['log_OR'], row[1]['log_p']
            formatted_label = insert_newline_after_third_space(row[1]['description'])
            ann = ax.annotate(
                formatted_label, 
                xy=(x, y),  
                xytext=(constant_x, label_y),  
                horizontalalignment="left", fontsize=5.5,
                arrowprops=dict(arrowstyle="-", color='black', linewidth=0.5, 
                                connectionstyle="angle,angleA=0,angleB=90,rad=0")  
            )
            annotations.append(ann)

        # **Style adjustments**
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Get current yticks and make them absolute
        yticks = ax.get_yticks()
        ax.set_yticklabels([str(int(abs(tick))) for tick in yticks])

        # Make spines thinner
        for spine in ax.spines.values():
            spine.set_linewidth(spine.get_linewidth() * 0.92)

        # Adjust axis padding and ticks
        ax.xaxis.labelpad = 2
        ax.tick_params(axis='y', labelsize=7, length=3)
        ax.tick_params(axis='x', labelsize=7, length=3)

        # **Label axes**
        ax.set_xlabel("log OR", fontsize=9)
        if j == 0:
            ax.set_ylabel("-log10(p-value)", fontsize=8)

        # **Add model title only to left plots**
        if j == 0:
            ax.set_title(f"{model}", fontsize=9, fontweight="bold", loc="left", y=1.02)
        
        ax.text(
            XLIM[0] + 0.4,  # Slightly left of the y-axis
            0.5 * ax.get_ylim()[1],  # near top
            cohort_map_box[cohort_top],  # use the pretty mapped name
            ha='right', va='center', fontsize=5, rotation=90  # <- rotation 90 degrees
        )
        ax.text(
            XLIM[0] +0.4,
            0.5 * ax.get_ylim()[0],  # near bottom
            cohort_map_box[cohort_bottom],
            ha='right', va='center', fontsize=5, rotation=90)

# Define legend handles for the cohorts
legend_elements = [
    mpatches.Patch(color=cohort_colors["YNHH"], label="Yale New Haven Hospitals"),
    mpatches.Patch(color=cohort_colors["Community"], label="Community Hospitals"),
    mpatches.Patch(color=cohort_colors["NEMG"], label="Outpatient Clinics"),
    mpatches.Patch(color=cohort_colors["UKB"], label="UK Biobank"),
]

fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.55, 0.9), frameon=False, fontsize=7, ncol=4)

save_path = os.path.join(base_path, "volcano_exp_grid.pdf")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

# %% Distribution

df = pd.read_csv(os.path.join(project.tabs_path, 'exp_preds_ynhhs.csv'))
# %%
# %%
# Your prediction columns
pred_cols = [
    'preds_image_EvenMonth',
    'preds_Dermatophytosis',
    'preds_Bitten_by_dog',
    'preds_Fractures',
    'preds_Headache',
    'preds_Transport_accident',
    'preds_Viral_respiratory_infection'
]

# Grid size: enough rows to fit all plots, 3 per row
n_cols = 3
n_rows = -(-len(pred_cols) // n_cols)  # ceiling division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(pred_cols):
    sns.histplot(df[col], bins=30, kde=True, ax=axes[i])
    axes[i].set_title(col, fontsize=12)
    axes[i].set_xlabel("Probability")
    axes[i].set_ylabel("Count")
    axes[i].set_xlim(0, 1)  # lock x-axis to [0, 1]

# Hide any unused subplots
for j in range(len(pred_cols), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()
