"""Scatterplots for comparing the different cohorts"""

# %%
import numpy as np 
import pandas as pd
import os
import project_constants as project
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import logit
from matplotlib.patches import Patch
from scipy.stats import pearsonr

#%% Define the file path
tabs_path = project.tabs_path  # Make sure this variable is correctly defined

# Define cohort names
cohorts = ["YNHH", "Community", "NEMG"]

# Define the datasets per condition
data_dict_yale = {
    "Sex": "yale_results_preds_image_MaleSex_{}_min20.csv",
    "LVSD": 'yale_results_preds_image_Under40_{}_min20.csv',
    "AS": 'yale_results_preds_image_ModerateOrSevereAS_{}_min20.csv',
    "MR": 'yale_results_preds_image_ModerateOrSevereMR_{}_min20.csv',
    "LVH": "yale_results_preds_image_HCM_LVDD_IVSd15_IntermediateAsFalse_{}_min20.csv",
    "SHD": "yale_results_preds_image_cmp_HCM_LVDD_ModtoSevVD_EF40_{}_min20.csv",
}

data_dict_ukb = {
    "Sex": "results_preds_male_sex_model_unfrozen_ep5_min20.csv",
    "LVSD": 'results_preds_image_Under40_min20.csv',
    "AS": 'results_preds_image_ModerateOrSevereAS_min20.csv',
    "MR": 'results_preds_image_ModerateOrSevereMR_min20.csv',
    "LVH": "results_preds_image_HCM_LVDD_IVSd15_IntermediateAsFalse_min20.csv",
    "SHD": "results_xgb_pred_cmp_HCM_LVDD_ModtoSevVD_EF40_min20.csv",
}

cohort_map_box = {
    "YNHH": " Yale New Haven Hospital",
    "UKB": "UK Biobank",
    "NEMG": "Outpatient clinics",
    "Community": "Community hospitals"
}

custom_palette = {
    "circulatory system": (0.984, 0.502, 0.447),
}

df_dict = {}

# Process both UKB and YALE files
for name in data_dict_ukb.keys():  # Assuming both dicts have the same keys
    df_list = []
    print(f"Loading data for: {name}")

    # Load UKB data (No {} formatting needed)
    if name in data_dict_ukb:
        file_path_ukb = os.path.join(tabs_path, data_dict_ukb[name])
        if os.path.exists(file_path_ukb):
            df_ukb = pd.read_csv(file_path_ukb)
            df_ukb['model'] = name
            df_ukb['cohort'] = 'UKB'
            df_list.append(df_ukb)
        else:
            print(f"File not found: {file_path_ukb}")

    # Load YALE data (Format the {} placeholder with the cohort name)
    for cohort in cohorts:  # Loop over YNHH, Community, NEMG
        if name in data_dict_yale:
            file_path_yale = os.path.join(tabs_path, data_dict_yale[name].format(cohort))
            if os.path.exists(file_path_yale):
                df_yale = pd.read_csv(file_path_yale)
                df_yale['model'] = name
                df_yale['cohort'] = cohort  # Use actual cohort name
                df_list.append(df_yale)
            else:
                print(f"File not found: {file_path_yale}")

    # Store the concatenated dataframe in the dictionary
    if df_list:
        df_dict[name] = pd.concat(df_list, ignore_index=True)

# Merge all conditions into a single dataframe
df_combined = pd.concat(list(df_dict.values()), ignore_index=True)

# %%
# Load and process data
df_combined = pd.concat(list(df_dict.values()), ignore_index=True)

# Pivot the data to prepare for plotting
df_pivot = df_combined.pivot_table(index=['description', 'model', 'group'], columns='cohort', values='OR').reset_index()

# Define cohort comparisons
cohort_comparisons = ["Community", "NEMG", "UKB"]
models = list(data_dict_ukb.keys())

# **Loop over each cohort comparison and create a separate figure**
for cohort in cohort_comparisons:
    # **Set up figure with subplots for each model (arrange in columns)**
    n_cols = min(3, len(models))  # Max 3 columns
    n_rows = int(np.ceil(len(models) / n_cols))  # Number of rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), sharex=True, sharey=True)

    # **Filter the data**
    df_filtered = df_pivot.dropna(subset=['YNHH', cohort])  # Remove missing values

    # **Remove outliers using the 90th percentile filter**
    Q90_X = np.percentile(df_filtered['YNHH'], 90)
    Q90_Y = np.percentile(df_filtered[cohort], 90)
    upper_bound_X = Q90_X * 1.5  
    upper_bound_Y = Q90_Y * 1.5  
    df_filtered = df_filtered[
        (df_filtered['YNHH'] <= upper_bound_X) & (df_filtered[cohort] <= upper_bound_Y)
    ]

    # **Compute axis limits**
    min_val = df_filtered[['YNHH', cohort]].min().min()
    max_val = df_filtered[['YNHH', cohort]].max().max()
    buffer = (max_val - min_val) * 0.1
    min_val, max_val = min_val - buffer, max_val + buffer

    # **Loop over models and plot**
    for idx, model in enumerate(models):
        col, row = divmod(idx, 2)  # Switch rows and columns (flip the indexing)
        ax = axes[row, col]  # Use 2D grid indexing

        # **Filter for specific model**
        df_model = df_filtered[df_filtered["model"] == model]

        # **Separate colored and gray points**
        is_colored = df_model["group"].isin(custom_palette.keys())  # Identify colored groups

        # **Plot gray points first (background)**
        ax.scatter(df_model.loc[~is_colored, "YNHH"], df_model.loc[~is_colored, cohort], 
                   c="gray", alpha=0.3, zorder=1)
        df_colored = df_model[df_model["group"].isin(custom_palette.keys())]

        # **Plot colored points on top**
        ax.scatter(df_model.loc[is_colored, "YNHH"], df_model.loc[is_colored, cohort], 
                   c=df_model.loc[is_colored, "group"].map(custom_palette), 
                   edgecolor='black', alpha=0.7, zorder=2)

        # **Reference line (y = x)**
        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", alpha=0.5)
        # Add dotted lines at x=1 and y=1
        ax.axhline(1, linestyle=":", color="black", linewidth=0.8)
        ax.axvline(1, linestyle=":", color="black", linewidth=0.8)
        
            # Pearson correlation

        # Pearson correlation only on cardiovascular phenotypes
        if len(df_colored) >= 2:
            r_val, p_val = pearsonr(df_colored["YNHH"], df_colored[cohort])
            p_val = project.sci_notation(float(p_val))
            ax.text(
                0.05, 0.95,
                f"r = {r_val:.2f}\np = {p_val}",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.7)
            )

        # **Set title for each subplot**
        ax.set_title(model, fontsize=10, fontweight="bold")

        # ✅ **Fix Y-label condition (Only for first column)**
        if col == 0:  
            ax.set_ylabel(f"Odds Ratio\n{cohort_map_box[cohort]}", fontsize=9)

        # ✅ **Fix X-label condition (Only for last row)**
        if row == n_rows - 1:  
            ax.set_xlabel("Odds Ratio\nYale New Haven Hospital", fontsize=9)
            
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    # **Remove empty subplots if models count is odd**
    for idx in range(len(models), n_rows * n_cols):  # Fill all grid spaces
        i, j = divmod(idx, n_cols)
        fig.delaxes(axes[i, j])  # Remove unused subplots

    # **Add a legend for the color mapping**
    legend_elements = [Patch(color=color, label=group) for group, color in custom_palette.items()]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False, fontsize=9)

    # **Adjust layout**
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space for legend

    # **Save the figure**
    output_file = os.path.join(project.tabs_path, f"scatter_YNHH_vs_{cohort}.pdf")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_file}")

    plt.show()
# %%
