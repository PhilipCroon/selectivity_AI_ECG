"Figures for the paper"


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
import matplotlib.ticker as mticker
from matplotlib.ticker import FixedLocator

# %%
# Define base path
base_path = "/home/pmc57/PheWas_AI_ECG/tabs"

# Define the cohort names
cohorts = ["YNHH", "Community", "NEMG"]

# Define the datasets per condition
data_dict_yale = {
            "Sex": "yale_results_preds_image_MaleSex_{}_min20.csv",
         "LVSD":'yale_results_preds_image_Under40_{}_min20.csv',
         "AS":'yale_results_preds_image_ModerateOrSevereAS_{}_min20.csv',
         "MR": 'yale_results_preds_image_ModerateOrSevereMR_{}_min20.csv',
         "LVH": "yale_results_preds_image_HCM_LVDD_IVSd15_IntermediateAsFalse_{}_min20.csv",
         "SHD": "yale_results_preds_image_cmp_HCM_LVDD_ModtoSevVD_EF40_{}_min20.csv",
     }

cohort_map = {
    "YNHH": " \nYale New Haven\nHospital",
    "UKB": "UK Bio\n ",
    "NEMG": "Outpatient clinics",
    "Community": "Community hospitals"
}

cohort_map_box = {
    "YNHH": " Yale New Haven Hospital",
    "UKB": "UK Bio",
    "NEMG": "Outpatient clinics",
    "Community": "Community hospital"
}  

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

# Function to plot pie chart
def plot_pie_chart(ax, df, title):
    """Plots a pie chart showing group proportions for the given dataset with a fixed order of categories."""

    # Ensure all groups are represented, even if they have zero count
    group_counts = df['group'].value_counts().reindex(group_order, fill_value=0)

    # Get colors in the correct order
    colors = [group_colors_ordered[group] for group in group_order]

    labels = [
        f"{group_counts[group] / group_counts.sum() * 100:.1f}%" if group == 'circulatory system' else "" 
        for group in group_order
    ]

    # Plot pie chart
    ax.pie(
        group_counts, 
        radius=1, 
        labels=labels, 
        labeldistance=0.45, 
        colors=colors, 
        wedgeprops=dict(width=0.6, edgecolor='w')  # Thicker pie chart
    )

    # Set title
    ax.set_title(title, fontsize=16, y=0.9)

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

    # Process merged data
    all_sig_list = list(set(all_sig_list))
    merged_data_all = pd.concat(data_list) if data_list else pd.DataFrame()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(data_list2) + 1, figsize=(28, 4))  # Each dataset + 1 for expected proportions

    # Loop through datasets to plot observed proportions
    for i, df in enumerate(data_list2):
        plot_pie_chart(axes[i+1], df, df['snp'].unique()[0])

    # Compute **expected proportions for this cohort only**
    if all_data_list:
        plot_pie_chart(axes[0], merged_data_all, 'Expected')

    axes[0].set_ylabel(cohort_map[cohort], fontsize=16, fontweight='bold', labelpad=10)

    # Adjust layout and save figure
    fig.subplots_adjust(hspace=0, wspace=-0.4)    
    plt.savefig(os.path.join(base_path, f'PheWAS_piecharts_yale_{cohort}.pdf'), dpi=600, bbox_inches='tight')
    plt.show()
    
    # Assuming `data_list2` is the list of dataframes, each containing 'snp', 'group', and 'bonferroni' columns.
    results = []
    total_tests = 0  # Initialize counter for the total number of tests

    # Loop through each dataframe in `data_list2`, representing a specific 'snp' or model
    for df in all_data_list: #skipping the first one bc it is combined
        snp_name = df['snp'].iloc[0]  # Get the name of the model from the 'snp' column
        group_counts_total = df['group'].value_counts(normalize=True)  # Get overall proportions of each group in this model
        group_counts_significant = df[df['p'] < sig_threshold]['group'].value_counts(normalize=True)  # Proportions among significant only

        # Iterate over each group to calculate observed vs expected and compute OR and p-value
        for group in group_counts_total.index:
            # Observed proportion (among significant observations)
            observed_proportion = group_counts_significant.get(group, 0)

            # Expected proportion (in the entire dataset)
            expected_proportion = group_counts_total[group]

            # Convert proportions to counts
            observed_count = observed_proportion * len(df[df['p'] < sig_threshold])  # Convert to count in significant set
            expected_count = expected_proportion * len(df)  # Convert to count in total set

            # Create a contingency table for Fisher's exact test
            contingency_table = [
                [observed_count, len(df[df['p'] < sig_threshold]) - observed_count],  # significant group
                [expected_count, len(df) - expected_count]  # all groups
            ]

            # Calculate odds ratio and p-value only if all values in the contingency table are non-zero
            try:
                odds_ratio, p_value = fisher_exact(contingency_table)
            except ValueError:
                odds_ratio, p_value = float('nan'), float('nan')

            # Append the results to the list
            results.append({
                'snp': snp_name,
                'group': group,
                'observed_proportion': observed_proportion,
                'expected_proportion': expected_proportion,
                'observed_count': observed_count,
                'expected_count': expected_count,
                'OR': odds_ratio,
                'p': p_value  # Store the unadjusted p-value
            })
            total_tests += 1  # Increment the test counter

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    # Maybe add to appendix later??
    print(results_df)
    results_df.to_csv(os.path.join(base_path, f'results_obs_exp_{cohort}.csv'))

    unique_snps = results_df['snp'].unique()
    num_snps = len(unique_snps)

    # Define a consistent y-axis range with additional padding
    y_min, y_max = 0, 8  # Slightly reduce y_min and increase y_max to add space
    # y_min, y_max = 0, results_df['OR'].max()*1.1
    yticks = np.arange(y_min, y_max + 1, 2)
    
    # Set up a grid of subplots in a single row with individual y-axes
    fig, axes = plt.subplots(1, len(unique_snps), figsize=(1.2 * num_snps, 2), sharey=False)

    for i, snp in enumerate(unique_snps):
        # Filter data for the specific SNP model and exclude zero values
        sub_data = results_df[(results_df['snp'] == snp) & (results_df['OR'] > 0)].copy()

        # Apply log transformation to OR values
    #     sub_data['log_OR'] = np.log10(sub_data['OR'])

        # Get unique groups and create evenly spaced x positions
        unique_groups = sub_data['group'].unique()
        x_positions = np.linspace(0.1, 0.9, len(unique_groups))  # Evenly spaced positions along x-axis
        group_positions = dict(zip(unique_groups, x_positions))  # Map each group to an x position

        # Assign each group its x position
        sub_data['x_position'] = sub_data['group'].map(group_positions)

        # Set alternating background colors
        background_color = 'whitesmoke' if i % 2 == 0 else 'white'
        axes[i].set_facecolor(background_color)

        significance_threshold = 0.05 / len(unique_snps)
       # Define marker types and sizes
        sub_data["marker"] = np.where(sub_data["p"] < significance_threshold, "d", "o")  # Square for significant, circle otherwise
        sub_data["size"] = np.where(sub_data["p"] < significance_threshold, 100, 50)  # Larger for significant, smaller otherwise

        # Plot non-significant points (circles)
        sns.scatterplot(
            data=sub_data[sub_data["p"] >= significance_threshold],
            x="x_position",
            y="OR",
            hue="group",
            palette=group_colors,
            style="marker",
            markers={"o": "o"},
            s=50,  # Small size for non-significant
            alpha=0.7,
            ax=axes[i]
        )

        # Plot significant points (squares) on top
        sns.scatterplot(
            data=sub_data[sub_data["p"] < significance_threshold],
            x="x_position",
            y="OR",
            hue="group",
            palette=group_colors,
            style="marker",
            markers={"d": "d"},
            s=70,  # Larger size for significant
            alpha=0.9,
            ax=axes[i]
        )
        
        axes[i].set_ylabel("")

        # Set consistent y-axis limits with padding
        axes[i].set_ylim(y_min, y_max+2)
        
        # Set x-axis limits from 0 to 1 to maintain spacing
        axes[i].set_xlim(0, 1)
        axes[i].set_xticks([])  # Remove x-ticks for all subplots
        
        # Turn off the legend for this subplot
        axes[i].legend([], [], frameon=False)
        
        # Custom y-ticks and y-label for the first plot only
        if i == 0:
            axes[i].set_ylabel('OR', fontsize=6)
            axes[i].set_yticks(np.arange(int(y_min), int(y_max) + 1, 2)) 
            axes[i].set_yticklabels([str(j) for j in range(int(y_min), int(y_max) + 1, 2)])
        else:
            axes[i].set_yticks([])  # Remove y-ticks from other subplots
            axes[i].spines['left'].set_visible(False)  # Remove left spine from other subplots

        # Remove top and right spines for a cleaner look
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

        # Set SNP name as x-axis label below each plot
        axes[i].set_xlabel(snp, fontsize=6)
        axes[i].axhline(1, color='gray', linestyle=':', linewidth=0.5)  # Dotted line at y=0
    # Create a custom legend on the right side of the figure
    handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.9, 0.5), ncol=1, frameon=False, fontsize=6)

    # Make all spines thinner
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)  # Set spine (border) thickness to 0.5

    # Make y-ticks thinner
    for ax in axes:
        ax.tick_params(axis='y', width=0.5)  # Set y-tick thickness to 0.5

    # Make y-tick labels smaller
    for ax in axes:
        ax.tick_params(axis='y', labelsize=5)  # Set y-tick label size to 'small'
        
    # axes[0].set_title(cohort_map_box[cohort], fontsize=5, x=0.1, y=0.9, fontweight='bold', ha='left', zorder=1000)
    fig.suptitle(cohort_map_box[cohort], fontsize=5, fontweight='bold', x=0.128, y=0.88, ha='left')
    # Adjust layout and remove space between subplots
    plt.subplots_adjust(wspace=0)

    # Save the plot
    plt.savefig(os.path.join(base_path, f'yale_PheWAS_boxplot_yale_{cohort}.pdf'), dpi=600, bbox_inches='tight')
    plt.show()

    # %%
    
#%% Piecharts UKB

data_dict_UKB = {
    "Sex": "results_preds_male_sex_model_unfrozen_ep5_min20.csv",
     "LVSD":'results_preds_image_Under40_min20.csv',
     "AS":'results_preds_image_ModerateOrSevereAS_min20.csv',
     "MR": 'results_preds_image_ModerateOrSevereMR_min20.csv',
     "LVH": "results_preds_image_HCM_LVDD_IVSd15_IntermediateAsFalse_min20.csv",
     "SHD": "results_xgb_pred_cmp_HCM_LVDD_ModtoSevVD_EF40_min20.csv",

 }

# Load the cohort-specific files
all_data_list = []
data_list = []
data_list2 = []
all_sig_list = []
for key, filename_template in data_dict_UKB.items():
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

# Create figure with subplots
fig, axes = plt.subplots(1, len(data_list2) + 1, figsize=(28, 4))  # Each dataset + 1 for expected proportions

# Loop through datasets to plot observed proportions
for i, df in enumerate(data_list2):
    plot_pie_chart(axes[i+1], df, df['snp'].unique()[0])

# Compute **expected proportions for this cohort only**
if all_data_list:
    plot_pie_chart(axes[0], merged_data_all, 'Expected')

axes[0].set_ylabel("UK Biobank", fontsize=16, fontweight='bold', labelpad=10, zorder=10)
# Add line between Sex and rest


# Adjust layout and save figure
fig.subplots_adjust(hspace=0, wspace=-0.4)    
plt.savefig(os.path.join(base_path, f'PheWAS_piecharts_UKB.pdf'), dpi=600, bbox_inches='tight')
plt.show()

# Assuming `data_list2` is the list of dataframes, each containing 'snp', 'group', and 'bonferroni' columns.
results = []
total_tests = 0  # Initialize counter for the total number of tests

# Loop through each dataframe in `data_list2`, representing a specific 'snp' or model
for df in all_data_list: #skipping the first one bc it is combined
    snp_name = df['snp'].iloc[0]  # Get the name of the model from the 'snp' column
    group_counts_total = df['group'].value_counts(normalize=True)  # Get overall proportions of each group in this model
    group_counts_significant = df[df['p'] < sig_threshold]['group'].value_counts(normalize=True)  # Proportions among significant only

    # Iterate over each group to calculate observed vs expected and compute OR and p-value
    for group in group_counts_total.index:
        # Observed proportion (among significant observations)
        observed_proportion = group_counts_significant.get(group, 0)

        # Expected proportion (in the entire dataset)
        expected_proportion = group_counts_total[group]

        # Convert proportions to counts
        observed_count = observed_proportion * len(df[df['p'] < sig_threshold])  # Convert to count in significant set
        expected_count = expected_proportion * len(df)  # Convert to count in total set

        # Create a contingency table for Fisher's exact test
        contingency_table = [
            [observed_count, len(df[df['p'] < sig_threshold]) - observed_count],  # significant group
            [expected_count, len(df) - expected_count]  # all groups
        ]

        # Calculate odds ratio and p-value only if all values in the contingency table are non-zero
        try:
            odds_ratio, p_value = fisher_exact(contingency_table)
        except ValueError:
            odds_ratio, p_value = float('nan'), float('nan')

        # Append the results to the list
        results.append({
            'snp': snp_name,
            'group': group,
            'observed_proportion': observed_proportion,
            'expected_proportion': expected_proportion,
            'observed_count': observed_count,
            'expected_count': expected_count,
            'OR': odds_ratio,
            'p': p_value  # Store the unadjusted p-value
        })
        total_tests += 1  # Increment the test counter

# Convert results to a DataFrame
results_df = pd.DataFrame(results)
# Maybe add to appendix later??
print(results_df)
results_df.to_csv(os.path.join(base_path, 'results_obs_exp_UKB.csv'))

unique_snps = results_df['snp'].unique()
num_snps = len(unique_snps)

# Define a consistent y-axis range with additional padding
y_min, y_max = 0, results_df['OR'].max()*1.1
yticks = np.arange(y_min, y_max + 1, 2)

# Set up a grid of subplots in a single row with individual y-axes
fig, axes = plt.subplots(1, len(unique_snps), figsize=(1.2 * num_snps, 2), sharey=False)

for i, snp in enumerate(unique_snps):
    # Filter data for the specific SNP model and exclude zero values
    sub_data = results_df[(results_df['snp'] == snp) & (results_df['OR'] > 0)].copy()

    # Get unique groups and create evenly spaced x positions
    unique_groups = sub_data['group'].unique()
    x_positions = np.linspace(0.1, 0.9, len(unique_groups))  # Evenly spaced positions along x-axis
    group_positions = dict(zip(unique_groups, x_positions))  # Map each group to an x position

    # Assign each group its x position
    sub_data['x_position'] = sub_data['group'].map(group_positions)

    # Set alternating background colors
    background_color = 'whitesmoke' if i % 2 == 0 else 'white'
    axes[i].set_facecolor(background_color)

        # Define marker types and sizes
    sub_data["marker"] = np.where(sub_data["p"] < significance_threshold, "d", "o")  # Square for significant, circle otherwise
    sub_data["size"] = np.where(sub_data["p"] < significance_threshold, 100, 50)  # Larger for significant, smaller otherwise

    # Plot non-significant points (circles)
    sns.scatterplot(
        data=sub_data[sub_data["p"] >= significance_threshold],
        x="x_position",
        y="OR",
        hue="group",
        palette=group_colors,
        style="marker",
        markers={"o": "o"},
        s=50,  # Small size for non-significant
        alpha=0.7,
        ax=axes[i]
    )

    # Plot significant points (squares) on top
    sns.scatterplot(
        data=sub_data[sub_data["p"] < significance_threshold],
        x="x_position",
        y="OR",
        hue="group",
        palette=group_colors,
        style="marker",
        markers={"d": "d"},
        s=70,  # Larger size for significant
        alpha=0.9,
        ax=axes[i]
    )
    
    axes[i].set_ylabel("")

    # Set consistent y-axis limits with padding
    axes[i].set_ylim(y_min, y_max)
    
    # Set x-axis limits from 0 to 1 to maintain spacing
    axes[i].set_xlim(0, 1)
    axes[i].set_xticks([])  # Remove x-ticks for all subplots
    
    # Turn off the legend for this subplot
    axes[i].legend([], [], frameon=False)
    
    # Custom y-ticks and y-label for the first plot only
    if i == 0:
        axes[i].set_ylabel('OR', fontsize=6)
        axes[i].set_yticks(np.arange(int(y_min), int(y_max) + 1, 2)) 
        axes[i].set_yticklabels([str(j) for j in range(int(y_min), int(y_max) + 1, 2)])
    else:
        axes[i].set_yticks([])  # Remove y-ticks from other subplots
        axes[i].spines['left'].set_visible(False)  # Remove left spine from other subplots

    # Remove top and right spines for a cleaner look
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)

    # Set SNP name as x-axis label below each plot
    axes[i].set_xlabel(snp, fontsize=6)
    axes[i].axhline(1, color='gray', linestyle=':', linewidth=0.5)  # Dotted line at y=0

# 1️⃣ **Create legend for group colors (squares)**
color_legend = [mpatches.Patch(color=color, label=group) for group, color in group_colors.items()]

# 2️⃣ **Create legend for significance markers (black)**
marker_legend = [
    Line2D([0], [0], marker='d', linestyle='None', color='black', markersize=6, label='Significant'),
    Line2D([0], [0], marker='o', linestyle='None', color='black', markersize=6, label='Non-significant')
]

# 3️⃣ **Combine both legends into one column**
legend_handles = color_legend + marker_legend

# **Position the legend on the right side**
fig.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(0.95, 0.5), 
           ncol=1, frameon=False, fontsize=6)

# Make all spines thinner
for ax in axes:
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)  # Set spine (border) thickness to 0.5

# Make y-ticks thinner
for ax in axes:
    ax.tick_params(axis='y', width=0.5)  # Set y-tick thickness to 0.5

# Make y-tick labels smaller
for ax in axes:
    ax.tick_params(axis='y', labelsize=5)  # Set y-tick label size to 'small'
    
# axes[0].set_title("UK Biobank", fontsize=5, x= 0.1, y=0.9, fontweight='bold', ha='left')
fig.suptitle("UK Biobank", fontsize=5, fontweight='bold', x=0.128, y=0.88, ha='left')

# Adjust layout and remove space between subplots
plt.subplots_adjust(wspace=0)

# Save the plot
plt.savefig(os.path.join(base_path, f'yale_PheWAS_boxplot_UKB.pdf'), dpi=600, bbox_inches='tight')
plt.show()


# %% barchart
# Filter data to your outcomes of interest
filtered_data = merged_data_total[
    merged_data_total['description'].isin(project.sig_list_curated_sub)
]

# Get unique cohorts and outcomes
cohorts = filtered_data['cohort'].unique()
outcomes = filtered_data['description'].unique()

n_rows = len(cohorts)         # 4
n_cols = len(outcomes)        # 5

# Create figure and axes
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), sharey='col')
axes = axes.reshape(n_rows, n_cols)

# Loop through each cohort (row) and outcome (column)
for i, cohort in enumerate(cohorts):
    for j, outcome in enumerate(outcomes):
        ax = axes[i, j]
        data = filtered_data[
            (filtered_data['cohort'] == cohort) &
            (filtered_data['description'] == outcome)
        ]

        if data.empty:
            ax.set_visible(False)
            continue

        # Plot ORs by model
        sns.barplot(
            data=data,
            x='snp',
            y='OR',
            ax=ax,
            palette='pastel'
        )

        # Style
        ax.axhline(1, linestyle='--', color='gray')
        ax.set_title(outcome, fontsize=9, fontweight='bold') if i == 0 else ax.set_title("")
        ax.set_ylabel(cohort if j == 0 else "")
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45, labelsize = 8)

# Improve layout
plt.tight_layout()
plt.show()

#%% clustered correlation heatmap: 
# Setup
darker_green = tuple([x * 0.7 for x in color_comm])

custom_cmap = LinearSegmentedColormap.from_list(
    'custom_diverging',
    [color_ukb, (1, 1, 1), darker_green],
    N=256
)


fig, axes = plt.subplots(2, 2, figsize=(8, 8))  # 2x2 grid, 4x4 inches per subplot
axes = axes.flatten()

for idx, cohort in enumerate(cohorts):
    ax = axes[idx]
    
    pivot = filtered_data[filtered_data['cohort'] == cohort].pivot(
        index='description', columns='snp', values='OR'
    )
    pivot.columns.name = None
    pivot.index.name = None

    if pivot.shape[1] < 2:
        ax.set_visible(False)
        continue

    corr = pivot.corr()
    
    print(corr)

    sns.heatmap(
    corr,
    ax=ax,
    annot=True,
    cmap=custom_cmap,  
    fmt=".2f",
    vmin=-0,
    vmax=1,
    center=0,
    cbar=False
)
    
    # Get the heatmap's color mesh for the colorbar
    color_mesh = [obj for obj in ax.get_children() if isinstance(obj, matplotlib.collections.QuadMesh)][0]

    # Add a custom small colorbar
    cbar = fig.colorbar(
        color_mesh,
        ax=ax,
        orientation='vertical',
        shrink=0.4,    # Adjust this for size
        pad=0.02       # Adjust distance from heatmap
    )
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Correlation", fontsize=9)
    
    ax.set_title(project.cohort_map_box[cohort], fontsize=10)
    ax.set_xlabel("Model", fontsize=10)
    ax.set_ylabel("Model", fontsize=10)

# Hide unused axes (if fewer than 4 cohorts)
for i in range(len(cohorts), 4):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig((os.path.join(project.tabs_path, 'Correlation_heatmap.pdf')))
plt.show()

#%% clustered correlation heatmap all phenotypes: 

filtered_data = merged_data_total[
    merged_data_total['group'] == "circulatory system"
]

darker = tuple([x * 0.7 for x in color_nemg])

custom_cmap = LinearSegmentedColormap.from_list(
    'custom_diverging',
    [color_ukb, (1, 1, 1), darker],
    N=256
)

fig, axes = plt.subplots(2, 2, figsize=(8, 8)) 
axes = axes.flatten()

for idx, cohort in enumerate(cohorts):
    ax = axes[idx]
    
    pivot = filtered_data[filtered_data['cohort'] == cohort].pivot(
        index='description', columns='snp', values='OR'
    )
    pivot.columns.name = None
    pivot.index.name = None

    if pivot.shape[1] < 2:
        ax.set_visible(False)
        continue

    corr = pivot.corr().round(2)

    sns.heatmap(
    corr,
    ax=ax,
    annot=True,
    cmap=custom_cmap, 
    fmt=".2f",
    vmin=0,
    vmax=1,
    center=0,
    cbar=False
)
    
    # Get the heatmap's color mesh for the colorbar
    color_mesh = [obj for obj in ax.get_children() if isinstance(obj, matplotlib.collections.QuadMesh)][0]

    # Add a custom small colorbar
    cbar = fig.colorbar(
        color_mesh,
        ax=ax,
        orientation='vertical',
        shrink=0.4,    
        pad=0.02       
    )
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Correlation", fontsize=9)
    
    ax.set_title(project.cohort_map_box[cohort], fontsize=10)
    ax.set_xlabel("Model", fontsize=10)
    ax.set_ylabel("Model", fontsize=10)

# Hide unused axes (if fewer than 4 cohorts)
for i in range(len(cohorts), 4):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig((os.path.join(project.tabs_path, 'Correlation_heatmap_full.pdf')))
plt.show()

#%% Heatmap
# Filter and get cohorts

# Ordered outcome names (columns), using original terms
ordered_models = list(data_dict_yale.keys())

custom_cmap = LinearSegmentedColormap.from_list(
    'custom_diverging',
    [color_ukb , (1, 1, 1), color_ynhh],
    N=256
)


ordered_outcomes = [
    "Heart failure NOS",
    "Aortic valve disease",
    "Mitral valve disease",
    "Cardiomegaly",
]

outcome_label_map = {
    "Heart failure NOS": "HF",
    "Aortic valve disease": "AVD",
    "Mitral valve disease": "MVD",
    "Cardiomegaly": "LVH", 
}

filtered = merged_data_total[
    merged_data_total['description'].isin(project.sig_list_curated_sub)
]

fig, axes = plt.subplots(2, 2, figsize=(8, 8), 
                         gridspec_kw=dict(hspace=0.3, wspace=0.2)) 
axes = axes.flatten()

for idx, cohort in enumerate(cohorts):
    ax = axes[idx]
    
    df_cohort = filtered[filtered['cohort'] == cohort]
    heatmap_data = df_cohort.pivot_table(
        index='snp',
        columns='description',
        values='OR'
    ).reindex(index=ordered_models, columns=ordered_outcomes)

    # Apply label mapping
    heatmap_data.columns = [outcome_label_map.get(col, col) for col in heatmap_data.columns]
    
    heatmap_data.index.name = None
    heatmap_data.columns.name = None

    sns.heatmap(
        heatmap_data,
        cmap=custom_cmap,
        center=1.0,
        vmin=-1,
        vmax=5,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        linecolor='gray',
        cbar=False,
        square=True,
        ax=ax,
        annot_kws={"size": 7}
    )
    
    # Get the heatmap's color mesh for the colorbar
    color_mesh = [obj for obj in ax.get_children() if isinstance(obj, matplotlib.collections.QuadMesh)][0]

    # Add a custom small colorbar
    cbar = fig.colorbar(
        color_mesh,
        ax=ax,
        orientation='vertical',
        shrink=0.4,   
        pad=0.02   
    )
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Odds Ratio", fontsize=8)

    # Title mapping and styling
    ax.set_title(project.cohort_map_box.get(cohort, cohort), fontsize=9, fontweight='normal')
    
    ax.set_xlabel("Phenotype", fontsize=8)
    ax.set_ylabel("Model", fontsize=8)
    

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='right', fontsize=6)
    ax.tick_params(axis='x', which='both', length=2, labelsize = 7)


    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
    ax.tick_params(axis='y', which='both', length=2, labelsize = 7)


# Hide unused axes
for i in range(len(cohorts), len(axes)):
    axes[i].axis('off')

plt.savefig(os.path.join(project.tabs_path, "ORs heatmap.pdf"))
plt.show()



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

# %% Volcano Combined

# **Define settings**
marker_size = 8
XLIM = [-3, 3]

# **Define cohort pairs for the single figure**
cohort_pair_sets = [
    ("YNHH", "Community"), 
    ("NEMG", "UKB")         
]

# **Get unique models from dataset**
unique_models = merged_data_total["snp"].unique()

# **Create the figure**
fig = plt.figure(figsize=(8, len(unique_models) * 4))  # Dynamic figure size
gs = gridspec.GridSpec(len(unique_models), 2, figure=fig, hspace=0.2, wspace=0.8)

# **Loop over each model to create individual plots in the grid**
for i, model in enumerate(unique_models):
    for j, (cohort_top, cohort_bottom) in enumerate(cohort_pair_sets):  # j=0 → Left, j=1 → Right
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
        
        # **Ensure p-values & OR are positive**
        data_top = data_top[(data_top["p"] > 0) & (data_top["OR"] > 0)]
        data_bottom = data_bottom[(data_bottom["p"] > 0) & (data_bottom["OR"] > 0)]

        # Step 3: Apply -log10 transformation after correction
        data_top["log_p"] = -np.log10(data_top["p"])
        data_bottom["log_p"] = -np.log10(data_bottom["p"])
        data_bottom["log_p"] = -data_bottom["log_p"]  # Flip for mirroring
        
        data_top["log_OR"] = np.log(data_top["OR"])
        data_bottom["log_OR"] = np.log(data_bottom["OR"])

        # **Determine y-axis limits**
        max_abs_y = max(data_top["log_p"].max(), abs(data_bottom["log_p"].min()))
        ax.set_ylim([-max_abs_y - 5, max_abs_y + 5])
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

# Define legend handles for the cohorts
legend_elements = [
    mpatches.Patch(color=cohort_colors["YNHH"], label="Yale New Haven Hospital"),
    mpatches.Patch(color=cohort_colors["Community"], label="Community Hospital"),
    mpatches.Patch(color=cohort_colors["NEMG"], label="Outpatient Clinics"),
    mpatches.Patch(color=cohort_colors["UKB"], label="UK Biobank"),
]

fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.55, 0.9), frameon=False, fontsize=7, ncol=4)

save_path = os.path.join(base_path, "volcano_combined_grid.pdf")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

# %%
# %% colcano Combined OR (no Log)

# **Define settings**
marker_size = 8
XLIM = [-3, 3]

# **Define cohort pairs for the single figure**
cohort_pair_sets = [
    ("YNHH", "Community"),  # Left column
    ("NEMG", "UKB")         # Right column
]

# **Get unique models from dataset**
unique_models = merged_data_total["snp"].unique()

# **Create the figure**
fig = plt.figure(figsize=(8, len(unique_models) * 4))  # Dynamic figure size
gs = gridspec.GridSpec(len(unique_models), 2, figure=fig, hspace=0.2, wspace=0.8)

# **Loop over each model to create individual plots in the grid**
for i, model in enumerate(unique_models):
    for j, (cohort_top, cohort_bottom) in enumerate(cohort_pair_sets):  # j=0 → Left, j=1 → Right
        ax = fig.add_subplot(gs[i, j])
        ax.axvline(1, color='black', linestyle='--', linewidth=0.8)
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
        
        # **Ensure p-values & OR are positive**
        data_top = data_top[(data_top["p"] > 0) & (data_top["OR"] > 0)]
        data_bottom = data_bottom[(data_bottom["p"] > 0) & (data_bottom["OR"] > 0)]

        # Step 3: Apply -log10 transformation after correction
        data_top["log_p"] = -np.log10(data_top["p"])
        data_bottom["log_p"] = -np.log10(data_bottom["p"])
        data_bottom["log_p"] = -data_bottom["log_p"]  # Flip for mirroring
        
        data_top["log_OR"] = np.log(data_top["OR"])
        data_bottom["log_OR"] = np.log(data_bottom["OR"])

        # **Determine y-axis limits**
        max_abs_y = max(data_top["log_p"].max(), abs(data_bottom["log_p"].min()))
        ax.set_ylim([-max_abs_y - 5, max_abs_y + 5])
        ax.set_xlim(XLIM)

        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)

        # **Find significant points**
        sig_top = data_top["p"] < stop
        sig_bottom = data_bottom["p"] < sbottom

        # **Plot non-significant points (Gray)**
        ax.scatter(data_top["OR"], data_top["log_p"], color="gray", alpha=0.3, s=marker_size)
        ax.scatter(data_bottom["OR"], data_bottom["log_p"], color="gray", alpha=0.3, s=marker_size)

        # **Plot significant points**
        ax.scatter(data_top.loc[sig_top, "OR"], data_top.loc[sig_top, "log_p"], 
                   color=cohort_colors[cohort_top], alpha=0.7, label=cohort_top, s=marker_size)

        ax.scatter(data_bottom.loc[sig_bottom, "OR"], data_bottom.loc[sig_bottom, "log_p"], 
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
        x_limits=(0.25, 4)
        ax.set_xscale('log')
        ax.set_xlim(x_limits)
        x_right = ax.get_xlim()[1] * (1.05)
        
        # Define ticks & labels
        or_ticks = [0.125, 0.25, 0.5, 1, 2, 4, 8]
        labels = ["⅛", "¼", "½", "1", "2", "4", "8"]

        # Set log scale & custom ticks
        ax.set_xscale("log")
        ax.set_xticks(or_ticks)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("Odds Ratio", fontsize=9)
        ax.xaxis.set_minor_locator(FixedLocator([]))

        # Optional: tighten the x-limits so ticks fit nicely
        ax.set_xlim(0.125, 8)
        # ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())

        constant_x = ax.get_xlim()[1] * 1.05  # Place labels outside the plot

        annotations = []

        # **Top Cohort Labels**
        for (row, label_y) in zip(top_6_TOP.iterrows(), y_positions_top):
            x, y = row[1]['OR'], row[1]['log_p']
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
            x, y = row[1]['OR'], row[1]['log_p']
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
        ax.set_xlabel("Odds Ratio", fontsize=9)
        if j == 0:
            ax.set_ylabel("-log10(p-value)", fontsize=8)

        # **Add model title only to left plots**
        if j == 0:
            ax.set_title(f"{model}", fontsize=9, fontweight="bold", loc="left", y=1.02)

# Define legend handles for the cohorts
legend_elements = [
    mpatches.Patch(color=cohort_colors["YNHH"], label="Yale New Haven Hospital"),
    mpatches.Patch(color=cohort_colors["Community"], label="Community Hospital"),
    mpatches.Patch(color=cohort_colors["NEMG"], label="Outpatient Clinics"),
    mpatches.Patch(color=cohort_colors["UKB"], label="UK Biobank"),
]

fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.55, 0.9), frameon=False, fontsize=7, ncol=4)

save_path = os.path.join(base_path, "volcano_combined_grid_OR.pdf")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()
# %%
