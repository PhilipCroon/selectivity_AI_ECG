"""
Bunch of forest plots

"""


# %%
import os
import sys
sys.path.append("/Users/philipcroon/PycharmProjects/CMR_gen_pred/")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import plot_misc.forest as forest
import plot_misc.heatmap as heatmap
import plot_misc.utils.utils as pm_utils
import plot_misc.example_data.examples as examples
%matplotlib inline
# from IPython.core.display import display
# import forestplot as fp
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter, NullFormatter
from matplotlib.ticker import LogLocator
from matplotlib.gridspec import GridSpec

import numpy as np
from scipy.stats import chi2
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform

import project_constants as project

def calculate_heterogeneity(or_vals, ci_lowers, ci_uppers):
    """Compute Cochran’s Q, I², and p-value from ORs and 95% CIs."""
    # Convert CIs to standard errors
    se = (np.log(ci_uppers) - np.log(ci_lowers)) / (2 * 1.96)
    weights = 1 / se**2
    log_ors = np.log(or_vals)

    weighted_mean = np.sum(weights * log_ors) / np.sum(weights)
    q = np.sum(weights * (log_ors - weighted_mean) ** 2)
    df = len(or_vals) - 1
    i2 = max(0, 100 * (q - df) / q) if q > df else 0
    p_value = 1 - chi2.cdf(q, df)

    return q, i2, p_value

# %%
# constants:
CMTOINCH = 1/2.54
sml = 0.5 # Create the main figure and subplots
fig_w = 17 * sml * CMTOINCH
row_h = sml * CMTOINCH
labelsize = 9*sml

# %%
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Formats for printing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def _superScriptinate(number):
    return number.replace('0','⁰').replace('1','¹').replace('2','²').\
        replace('3','³').replace('4','⁴').replace('5','⁵').replace('6','⁶')\
        .replace('7','⁷').replace('8','⁸').replace('9','⁹').replace('-','⁻')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def sci_notation(number:float, sig_fig:int=2,
                 max:float=np.float_power(10, -100)
                 ) -> str:
    """
    Returns a number in scientific notation with the lead numbers to  a
    specific significant number `sig_fig`
    
    Automatically truncates values if too small to print.
    """
    if number < max:
        number = max
    # getting string
    ret_string = "{0:.{1:d}e}".format(number, sig_fig)
    try:
        a,b = ret_string.split("e")
        # removed leading "+" and strips leading zeros too.
        b = int(b)
        return a + "×10" + _superScriptinate(str(b))
    except ValueError or TypeError:
        return str(np.nan)
    
# %%
tabs_path = project.tabs_path

# Define cohort names
cohorts = ["YNHH", "Community", "NEMG"]

file_dict_yale = { 'SHD':'yale_results_cox_noCMP_preds_image_cmp_HCM_LVDD_ModtoSevVD_EF40_{}.csv', 
 'LVSD': 'yale_results_cox_noCMP_preds_image_Under40_{}.csv', 
'MR': 'yale_results_cox_noCMP_preds_image_ModerateOrSevereMR_{}.csv', 
 'AS': 'yale_results_cox_noCMP_preds_image_ModerateOrSevereAS_{}.csv', 
 'LVH':'yale_results_cox_noCMP_preds_image_HCM_LVDD_IVSd15_IntermediateAsFalse_{}.csv', 
  'Sex': 'yale_results_cox_noCMP_preds_image_MaleSex_{}.csv'}

file_dict_ukb = { 'SHD':'results_cox_noCMP_xgb_pred_cmp_HCM_LVDD_ModtoSevVD_EF40.csv', 
 'LVSD': 'results_cox_noCMP_preds_image_Under40.csv', 
'MR': 'results_cox_noCMP_preds_image_ModerateOrSevereMR.csv', 
 'AS': 'results_cox_noCMP_preds_image_ModerateOrSevereAS.csv', 
 'LVH':'results_cox_noCMP_preds_image_HCM_LVDD_IVSd15_IntermediateAsFalse.csv', 
  'Sex': 'results_cox_noCMP_preds_image_MaleSex.csv'}

cohort_map_box = {
    "YNHH": " Yale New Haven Hospital",
    "UKB": "UK Biobank",
    "NEMG": "Outpatient clinics",
    "Community": "Community hospitals"
}

# Initialize dictionary to store dataframes per model
df_dict = {}

# Process both UKB and YALE files
for name in file_dict_ukb.keys():  # Assuming both dictionaries have the same keys
    df_list = []
    print(f"Processing: {name}")

    # Load UKB data (directly from file_dict_ukb)
    file_path_ukb = os.path.join(tabs_path, file_dict_ukb[name])
    if os.path.exists(file_path_ukb):
        df_ukb = pd.read_csv(file_path_ukb)
        df_ukb['model'] = name
        df_ukb['cohort'] = 'UKB'
        df_ukb['description'] = df_ukb['description'].map({
                            'Heart failure NOS': 'Heart failure',
                            'Cardiomegaly': 'Left ventricular hypertrophy'
                        }).fillna(df_ukb['description'])
    else:
        print(f"File not found: {file_path_ukb}")

    # Load Yale data (formatted with cohort names)
    for cohort in cohorts:
        if name in file_dict_yale:
            file_path_yale = os.path.join(tabs_path, file_dict_yale[name].format(cohort))
            if os.path.exists(file_path_yale):
                df_yale = pd.read_csv(file_path_yale)
                df_yale['model'] = name
                df_yale['cohort'] = cohort  # Assign cohort name
                df_yale['description'] = df_yale['description'].map({'Heart failure NOS': 'Heart failure',
                                                          'Cardiomegaly': "Left ventricular hypertrophy"
                                                          }).fillna(df_yale['description'])
                df_list.append(df_yale)
            else:
                print(f"File not found: {file_path_yale}")

    # Concatenate data for this model if any data exists
    df_list.append(df_ukb)
    if df_list:
        df_dict[name] = pd.concat(df_list, ignore_index=True)

# Concatenate all models into a single DataFrame
df_combined = pd.concat(df_dict.values(), ignore_index=True)

# Define refined colors for each cohort
color_ukb = (0.902, 0.298, 0.235)   # Warm coral red
color_ynhh = (0.050, 0.164, 0.376)  # Deep navy blue (Yale)
color_comm = (0.565, 0.741, 0.565)  # Soft sage green
color_nemg = (1.0, 1.0, 0.702)  # yellow 

color_dict1 = {
    "YNHH": color_ynhh,
    "Community": color_comm,
    "NEMG": color_nemg,
    "UKB": color_ukb,
}

color_dict3 = {
    "Yale New Haven Hospital": color_ynhh,
    "Community hospitals": color_comm,
    "Outpatient clinics": color_nemg,
    "UK Biobank": color_ukb,
}

cohort_map_box = {
    "YNHH": " Yale New Haven Hospital",
    "UKB": "UK Biobank",
    "NEMG": "Outpatient clinics",
    "Community": "Community hospitals"
}

color_dict2 = {
    'MR': '#F09EA7',       # Color 1
    'AS': '#FAFABE',       # Color 2
    'VHD': '#F6CA94',      # Color 3
    'LVSD': '#C1EBC0',     # Color 4
    'SHD': '#C7CAFF',      # Color 5
    'LVH': '#F6C2F3',      # Color 6
    'HCM': '#B0D2EB',      # Color 7
    'sex': '#F4A9C0'       # Color 8 
}

# %% For appendix including tables
# Constants for layout
row_h = 0.2  # Adjust as needed to get consistent row heights
p_threshold = 0.05 / 15  # Define the p-value threshold TODO: Check if still correct

#%% First loop: Process each dataframe
for indx, mods in enumerate([['Sex', 'LVSD'], ['AS', 'MR'], ['LVH', 'SHD']]):
    processed_data = []
    row_counts = []
    for mod in mods:
        data = df_combined[df_combined['model']== mod]
        # Apply color and shape mappings
        data['col'] = data['cohort'].map(color_dict1)
        data['shape'] = 'o'
        data['alpha'] = 1

        # Set index, reset, and format values
        data = data.set_index(['description']).fillna("").reset_index()
        data['p_value'] = data['p'].astype(float).apply(sci_notation)

        # Sort and assign distance for y-axis spacing
        var_list = data['description'].to_list()
        sort_dict = {value: index for index, value in enumerate(var_list)}

        # Convert columns to float for OR and CI
        data['OR'] = data['exp(coef)'].astype(float)
        data['Lower_CI'] = data['exp(coef) lower 95%'].astype(float)
        data['Upper_CI'] = data['exp(coef) upper 95%'].astype(float)
        data['string'] = data['OR'].map('{:,.2f}'.format) + ' (' + \
                         data['Lower_CI'].map('{:,.2f}'.format) + ';' + \
                         data['Upper_CI'].map('{:,.2f}'.format) + ')'

        # Filter data by p-value threshold and save filtered data
        data['p'] = data['p'].astype(float)

        data = forest.assign_distance(data, group='description', sort_dict=sort_dict, between_pad=6)
        processed_data.append(data)
        row_counts.append(len(data))

    # Calculate the number of rows in each dataframe
    num_rows = [len(df) for df in processed_data]

    # Calculate the total number of rows across all dataframes
    total_rows = sum(num_rows)

    # Calculate the height ratios based on the number of rows in each dataframe
    height_ratios = [(num_row / total_rows) for num_row in num_rows]

    # Set up the figure and axes using gridspec
    fig, ax = plt.subplots(len(processed_data), 3, figsize=(fig_w, row_h * total_rows*0.8),
                           gridspec_kw={'width_ratios': [10, 4, 4], 'height_ratios': height_ratios}, sharex='col')

    # Adjust the layout to remove unnecessary space between subplots
    plt.subplots_adjust(wspace=0.0, hspace=0.05)

    # Loop through the processed data with an index to place each plot in the correct subplot row
    for i, data in enumerate(processed_data):

        # Plot the forest plot on the corresponding subplot (row i, column 0)
        _, _, span = forest.plot_forest(df=data, x_col='OR', lb_col='Lower_CI', ub_col='Upper_CI', s_col='shape', c_col='col', ci_colour='black',
                                        g_col='description', s_size_col=50 * sml, ci_lwd=1.2,
                                        ax=ax[i, 0], span_return=True,
                                        kwargs_scatter_dict={'edgecolors': 'black', 'zorder': 3},
                                        kwargs_plot_ci_dict={'zorder': 2, 'solid_capstyle': 'round', 'linestyle': '-'})

        # Change format for each subplot (row i, column 0)
        ax[i, 0].set_title(data.loc[0, 'model'], fontsize=11*sml, loc='left', fontweight='bold', y=0.96, x=0.03)        
        ax[i, 0].spines[['right', 'left', 'top']].set_visible(False)
        ax[i, 0].spines[['left']].set_visible(True)
        ax[i, 0].axvline(1, linewidth=1, linestyle='--', c='black', zorder=2)
        ax[i, 0].yaxis.set_ticklabels(ax[i, 0].yaxis.get_ticklabels(), weight='bold', size=11 * sml)
        ax[i, 0].tick_params('y', labelsize=9 * sml, left=False)
        ax[i, 2].set_ylim(ax[i, 0].get_ylim())

        current_ylim = ax[i, 0].get_ylim()
        y_range = current_ylim[1] - current_ylim[0]  # Get the range of Y-axis

        new_lower_bound = current_ylim[0] - (y_range * 0.025)  # Lower by 2.5%
        new_upper_bound = current_ylim[1] + (y_range * 0.05)   # Increase upper limit by 5%

        # Apply to all subplots in the row
        ax[i, 0].set_ylim(new_lower_bound, new_upper_bound)
        ax[i, 1].set_ylim(new_lower_bound, new_upper_bound)
        ax[i, 2].set_ylim(new_lower_bound, new_upper_bound)

        # Define headers for the first row only
        if i == 0: 
            annote1 = 'HR\n(95% CI)'
            annote2 = 'p value'
            size_header = 9 * sml
            pad = 2
        else: 
            annote1 = None
            annote2 = None
            size_header = 0
            pad = -10

        # Plot the first table with OR and CI
        _ = forest.plot_table(data, annoteheader=annote1, string_col="string", ax=ax[i, 1],
                              halignment_text='center', halignment_header='center', pad_header=1, size_text=10 * sml,
                              size_header=size_header, negative_padding=pad)

        # Plot the second table with p-values
        _ = forest.plot_table(data, annoteheader=annote2, string_col="p_value", ax=ax[i, 2],
                              halignment_text='center', halignment_header='center', pad_header=1, size_text=10 * sml,
                              size_header=size_header, negative_padding=pad)

        # Customize each row and add background shading if needed
        span_span = span.span
        colors = [span_span[key]['kwargs']['color'] for key in span_span]
        colors.reverse()

        # Reassign the reversed colors back to the dictionary
        for key, color in zip(span.span, colors):
            span_span[key]['kwargs']['color'] = color
        span_span[0]['min'] = 0.0

        for index in range(len(ax[i])):
            for s in span_span:
                ax[i, index].axhspan(ymin=span_span[s]["min"], ymax=span_span[s]['max'], **span_span[s]["kwargs"])
        ax[i, 1].spines[['bottom']].set_visible(True)
        ax[i, 2].spines[['bottom']].set_visible(True)

        # Configure x-axis for the log scale
        ax[i, 0].set_xscale('log')
        for axis in [ax[i, 0].xaxis]:
            axis.set_major_formatter(ScalarFormatter())
            axis.set_minor_formatter(NullFormatter())
        ax[i, 0].tick_params(axis='both', which='minor', bottom=False, labelsize=4, width=0.5)

    # Configure x-axis labels for the last row
    ax[-1, 0].set_xlabel('Hazard Ratio (95% CI)', fontsize=11 * sml)
    ax[-1, 0].tick_params('x', labelsize=9 * sml, left=False)
    ax[-1, 0].set_xticks([0.5, 1, 2])
    ax[-1, 0].set_xticklabels(['0.5', '1', '2'])
    
    COL_val = list(color_dict3.values())
    COL_key = list(color_dict3.keys())
    leg_handles1 = []
    leg_handles2 = []


    for i in range(len(COL_key)):
        leg_handles2 = leg_handles2 + [
            Line2D([0], [0], marker='o', color=COL_val[i],
                   markersize=14*sml, lw=0,
                   label=COL_key[i],
                   markeredgecolor='black',
                   )
        ]

    legend_position2 = (-0.027, 1.05)

    # Calculate the legend position based on the number of rows
    legend2 = ax[0, 0].legend(handles=leg_handles2, ncol=4, bbox_to_anchor = legend_position2, loc="lower left", frameon = False, fontsize=4, markerscale=1, columnspacing=1.7)

#         ax[0, 0].add_artist(legend1)
    fig.add_artist(legend2)

    # Show or save the figure
    plt.savefig(os.path.join(tabs_path, f'forestplot_preds_{indx}.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
    
#%% Two columns for main paper

# Define figure layout: 2 columns, 3 rows
fig, ax = plt.subplots(2, 3, figsize=(7.5, row_h * total_rows*0.8), sharex=True, sharey=True)
    # Adjust the layout to remove unnecessary space between subplots
plt.subplots_adjust(wspace=-0.1, hspace=0.03)
# Loop over models and place them in columns
for i, mod in enumerate(['Sex', 'LVSD', 'AS', 'MR', 'LVH', 'SHD']):
    col = i // 2  # Determines left (0) or right (1) column
    row = i % 2   # Places model in correct row

    data = df_combined[df_combined['model'] == mod]
    print(data)

    # Apply color and shape mappings
    data['col'] = data['cohort'].map(color_dict1)
    # data['shape'] = np.where(data['p'] < 0.00714, 'd', 'o')
    data['shape'] = 'o'
    data['alpha'] = 1

    # Convert columns to float for OR and CI
    data['OR'] = data['exp(coef)'].astype(float)
    data['Lower_CI'] = data['exp(coef) lower 95%'].astype(float)
    data['Upper_CI'] = data['exp(coef) upper 95%'].astype(float)

    # Sort and assign spacing for y-axis
    var_list = data['description'].to_list()
    sort_dict = {value: index for index, value in enumerate(var_list)}
    data = forest.assign_distance(data, group='description', sort_dict=sort_dict, between_pad=6)

    # **Plot the forest plot**
    _, _, span = forest.plot_forest(
        df=data, x_col='OR', lb_col='Lower_CI', ub_col='Upper_CI', 
        s_col='shape', c_col='col', ci_colour='black',
        g_col='description', s_size_col=70 * sml, ci_lwd=1.2,
        ax=ax[row, col], span_return=True,
        kwargs_scatter_dict={'edgecolors': 'black', 'zorder': 3},
        kwargs_plot_ci_dict={'zorder': 2, 'solid_capstyle': 'round', 'linestyle': '-'}
    )

    # **Customize each subplot**
    ax[row, col].set_title(mod, fontsize=8, loc='left', fontweight='bold', y=0.94, x=0.03)        
    ax[row, col].spines[['right', 'top']].set_visible(False)
    ax[row, col].axvline(1, linewidth=1, linestyle='--', c='black', zorder=2)
    ax[row, col].set_xscale('log')

    # **Adjust Y-axis spacing**
    current_ylim = ax[row, col].get_ylim()
    y_range = current_ylim[1] - current_ylim[0]
    new_lower_bound = current_ylim[0] - (y_range * 0.025)  
    new_upper_bound = current_ylim[1] + (y_range * 0.05)   
    ax[row, col].set_ylim(new_lower_bound, new_upper_bound)

    # **Customize X-axis**
    ax[row, col].set_xticks([0.5, 1, 2])
    ax[row, col].set_xticklabels(['0.5', '1', '2'], fontsize=8)
    ax[row, col].xaxis.set_major_formatter(ScalarFormatter())
    ax[row, col].xaxis.set_minor_formatter(NullFormatter())
    ax[row, col].tick_params(axis='y', length=0, labelsize=8)
    
    # Remove X-labels for all but the last row
    if row != 1:
        ax[row, col].set_xticklabels([])
    else:
        ax[row, col].set_xlabel('Hazard Ratio (95% CI)', fontsize=8)
        
# Create legend handles based on color_dict1
legend_handles = [
    Line2D([0], [0], marker='o', color=color, markersize=8, lw=0, label=cohort_map_box[label], markeredgecolor='black')
    for label, color in color_dict1.items()
]

# Add the legend at the top of the figure
fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.93),
           ncol=len(color_dict1), frameon=False, fontsize=8, columnspacing=0.5)

# **Save the figure**
output_path = os.path.join(tabs_path, 'forestplot_combined.pdf')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()


#%% Pivot table: models as rows, phenotypes as columns
for cohort in cohorts:
    print(f"\n=== Cohort: {cohort} ===")

    # Subset for this cohort
    df_sub = df_combined[df_combined['cohort'] == cohort]

    # Pivot: models as rows, phenotypes as columns (handle duplicates with mean)
    pivot_df = df_sub.pivot_table(
        index='model',
        columns='description',
        values='exp(coef)',
        aggfunc='mean'
    )

    # Optional: drop phenotypes with missing data
    pivot_df = pivot_df.dropna(axis=1, how='any')

    # Log-transform ORs to linearize
    log_or_df = np.log(pivot_df)

    # Compute correlation
    corr_matrix = log_or_df.transpose().corr()

    # Plot correlation heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='vlag', vmin=-1, vmax=1, fmt=".2f",
                cbar_kws={'label': 'Pearson r'}, square=True)
    plt.title(f'Model Correlation in {cohort} Cohort')
    plt.tight_layout()
    plt.show()
    

# %% Forest sensitivity
df_sens = pd.read_csv(os.path.join(project.tabs_path, 'modelsxoutcomes.csv'))

print(df_sens)

# %%
df_lvef = df_sens[['model', 'LV ejection fraction_2.0']]
df_lvh = df_sens[['model', 'LVH_true']]


# %%
# normalize en-dash to hyphen, extract three numbers, convert to numeric
pat = r'([\d.]+)\s*\(\s*([\d.]+)-([\d.]+)\s*\)'
tmp = df_lvef['LV ejection fraction_2.0'].astype(str).str.replace('–', '-', regex=False)
parts = tmp.str.extract(pat)
df_lvef[['OR', 'CI_low', 'CI_high']] = parts.apply(pd.to_numeric, errors='coerce')

# normalize en-dash to hyphen, extract three numbers, convert to numeric
pat = r'([\d.]+)\s*\(\s*([\d.]+)-([\d.]+)\s*\)'
tmp = df_lvh['LVH_true'].astype(str).str.replace('–', '-', regex=False)
parts = tmp.str.extract(pat)
df_lvh[['OR', 'CI_low', 'CI_high']] = parts.apply(pd.to_numeric, errors='coerce')

# quick check
print(df_lvef[['model', 'OR', 'CI_low', 'CI_high']].head())
print(df_lvh[['model', 'OR', 'CI_low', 'CI_high']].head())

# %%
df_lvef2 = df_lvef.copy()
df_lvef2['evaluated_outcome'] = 'Left ventricular systolic dysfunction (CMR)'

df_lvh2 = df_lvh.copy()
df_lvh2['evaluated_outcome'] = 'Left ventricular hypertrophy (LVH)'

combined = pd.concat([df_lvef2, df_lvh2], ignore_index=True)
combined = combined[['model', 'evaluated_outcome', 'OR', 'CI_low', 'CI_high']]

models_to_keep = [
    'preds_image_HCM_LVDD_IVSd15_IntermediateAsFalse',
    'preds_image_ModerateOrSevereAS',
    'preds_image_ModerateOrSevereMR',
    'preds_image_Under40',
    'xgb_pred_cmp_HCM_LVDD_ModtoSevVD_EF40',
]

# 1) subset combined (assumes `combined` already exists with required columns)
combined = combined[combined['model'].isin(models_to_keep)].copy()


print(combined['model'].to_list())

combined['col'] = 'blueviolet'
combined['shape'] = 'o'
combined['model'] = combined['model'].map(project.preds_map_ukb)
print(combined['model'])

color_dict2 = {'Left ventricular systolic dysfunction (CMR)': 'blueviolet', 'Left ventricular hypertrophy (LVH)': 'orangered'}

combined['col'] = combined['evaluated_outcome'].map(color_dict2)
combined['shape'] = 'o'
combined['alpha'] = 1

# Sort and assign distance for y-axis spacing
var_list = set(combined['model'].to_list())
sort_dict = {value: index for index, value in enumerate(var_list)}

combined = forest.assign_distance(combined, group='model', sort_dict=sort_dict, between_pad=6)

# plot params
fig_w = 7
fig_h = max(4, len(combined['model'].unique()) * 0.35)
fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

# call forest.plot_forest (returns handles and span if span_return=True)
# adapt s_size_col to taste (size of markers)
_, _, span = forest.plot_forest(
    df=combined,
    x_col='OR',
    lb_col='CI_low',
    ub_col='CI_high',
    s_col='shape',
    c_col='col',
    ci_colour='black',
    g_col='model',
    s_size_col=80,
    ci_lwd=1.2,
    ax=ax,
    span_return=True,
    kwargs_scatter_dict={'edgecolors': 'black', 'zorder': 3},
    kwargs_plot_ci_dict={'zorder': 2, 'solid_capstyle': 'round', 'linestyle': '-'}
)

# cosmetics
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('AUROC (95% CI)', fontsize=11)
ax.tick_params(axis='both', labelsize=9)
ax.axvline(0.5, linewidth=1, linestyle='--', color='black', zorder=1)

# Attempt to set y-tick labels to model names (only if ticks length matches models)
models_order = list(dict.fromkeys(combined['model'].tolist()))
yticks = ax.get_yticks()
if len(yticks) == len(models_order):
    ax.set_yticklabels(models_order, fontsize=9)
else:
    # fallback: try using combined order reversed (forest often plots top->bottom)
    alt = models_order[::-1]
    if len(yticks) == len(alt):
        ax.set_yticklabels(alt, fontsize=9)

# build legend for evaluated_outcome colors (if you have color mapping dict)
# try to infer color mapping from combined DataFrame
color_map = dict(zip(combined['evaluated_outcome'], combined['col']))
# reduce to unique mapping preserving order
seen = {}
for k, v in color_map.items():
    if k not in seen:
        seen[k] = v
color_items = list(seen.items())

leg_handles = [
    Line2D([0], [0], marker='o', color=color, markersize=8, markeredgecolor='black', lw=0, label=label)
    for label, color in color_items
]
if leg_handles:
    ax.legend(handles=leg_handles, ncol=1, bbox_to_anchor=(0.1, 1.2), loc='upper left', frameon=True, fontsize='small')

plt.tight_layout()
plt.show()

# %%

# ensure numeric OR/CI columns (if not already)
for df in (df_lvef, df_lvh):
    for col in ['OR', 'CI_low', 'CI_high']:
        if col not in df.columns:
            raise KeyError(f"Missing column {col} in {df}")
        df[col] = pd.to_numeric(df[col], errors='coerce')

# prepare combined table and restrict to the models you want
models_to_keep = [
    'preds_image_HCM_LVDD_IVSd15_IntermediateAsFalse',
    'preds_image_ModerateOrSevereAS',
    'preds_image_ModerateOrSevereMR',
    'preds_image_Under40',
    'xgb_pred_cmp_HCM_LVDD_ModtoSevVD_EF40',
]

df_lvef2 = df_lvef.copy()
df_lvef2['evaluated_outcome'] = 'Left ventricular systolic dysfunction (CMR)'

df_lvh2 = df_lvh.copy()
df_lvh2['evaluated_outcome'] = 'Left ventricular hypertrophy (CMR)'

combined = pd.concat([df_lvef2, df_lvh2], ignore_index=True)
combined = combined[combined['model'].isin(models_to_keep)].copy()

# map model keys to pretty labels (presumes project.preds_map_ukb exists)
combined['model'] = combined['model'].map(lambda k: project.preds_map_ukb.get(k, k))

# color / shape
color_dict2 = {
    'Left ventricular systolic dysfunction (CMR)': 'blueviolet',
    'Left ventricular hypertrophy (CMR)': 'orangered'
}
combined['col'] = combined['evaluated_outcome'].map(color_dict2)
combined['shape'] = 'o'

# enforce a consistent model ordering (left->right plots will share this order)
models_order = [project.preds_map_ukb.get(k, k) for k in models_to_keep]

# create sort dict required by forest.assign_distance
sort_dict = {m: i for i, m in enumerate(models_order)}

# split subsets
left_outcome = 'Left ventricular systolic dysfunction (CMR)'
right_outcome = 'Left ventricular hypertrophy (CMR)'
df_left = combined[combined['evaluated_outcome'] == left_outcome].copy()
df_right = combined[combined['evaluated_outcome'] == right_outcome].copy()

# assign distances (same sort_dict so y-positions align across plots)
df_left = forest.assign_distance(df_left, group='model', sort_dict=sort_dict, between_pad=6)
df_right = forest.assign_distance(df_right, group='model', sort_dict=sort_dict, between_pad=6)

# plotting
n_models = len(models_order)
fig_h = max(4, n_models * 0.6)
fig, axes = plt.subplots(1, 2, figsize=(12, fig_h), sharey=True)

for ax, df_sub, title in [
    (axes[0], df_left, left_outcome),
    (axes[1], df_right, right_outcome),
]:
    _, _, span = forest.plot_forest(
        df=df_sub,
        x_col='OR',
        lb_col='CI_low',
        ub_col='CI_high',
        s_col='shape',
        c_col='col',
        ci_colour='black',
        g_col='model',
        s_size_col=80,
        ci_lwd=1.2,
        ax=ax,
        span_return=True,
        kwargs_scatter_dict={'edgecolors': 'black', 'zorder': 3},
        kwargs_plot_ci_dict={'zorder': 2, 'solid_capstyle': 'round', 'linestyle': '-'}
    )

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('AUROC (95% CI)', fontsize=10)
    ax.axvline(0.5, linewidth=1, linestyle='--', color='black', zorder=1)
    ax.tick_params(axis='x', labelsize=9)
    

# set shared y-tick labels using the consistent order (forest often plots top->bottom)
# get current y-ticks from left axis and map to models_order (reverse if necessary)
yticks = axes[0].get_yticks()
if len(yticks) == len(models_order):
    # forest plots top-to-bottom; reverse model list to match typical plotting order
    axes[0].set_yticklabels(models_order[::-1], fontsize=9)
else:
    # fallback: use the labels from the dataframe (maintain their plotted order)
    labels = df_left['model'].unique()[::-1]
    axes[0].set_yticklabels(labels, fontsize=9)

# legend for outcomes
handles = [
    Line2D([0], [0], marker='o', color=color_dict2[left_outcome], markersize=8, markeredgecolor='black', lw=0, label=left_outcome),
    Line2D([0], [0], marker='o', color=color_dict2[right_outcome], markersize=8, markeredgecolor='black', lw=0, label=right_outcome),
]

xt = np.arange(0.5, 1.01, 0.1)   # 0.5,0.6,...,1.0
xt = np.round(xt, 2)

for ax in axes:
    ax.set_xlim(0.5, 1.0)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{t:.1f}" for t in xt])
    ax.tick_params(axis='x', labelsize=9)

plt.tight_layout()
plt.show()

# %%
