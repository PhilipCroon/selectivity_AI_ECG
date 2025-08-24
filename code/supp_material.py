"""
Script to create the supplementary data files
"""

# %%
import numpy as np
import pandas as pd
import os
import project_constants as project

# 
tabs_path = project.tabs_path
save_path = os.path.join(tabs_path, "Supplementary_data.xlsx")

string_dict = {
    "YNHH": "The results of the fisher's exact test for enrichment of cardiac phenotypes in Yale New Haven Health",
    "Community": "The results of the fisher's exact test for enrichment of cardiac phenotypes in community hospitals",
    "NEMG": "The results of the fisher's exact test for enrichment of cardiac phenotypes in outpatient clinics",
    "UKB": "The results of the fisher's exact test for enrichment of cardiac phenotypes in the UK Biobank",
    
}

data_dict_yale = {
            "Sex": "yale_results_preds_image_MaleSex_{}_min20.csv",
         "LVSD":'yale_results_preds_image_Under40_{}_min20.csv',
         "AS":'yale_results_preds_image_ModerateOrSevereAS_{}_min20.csv',
         "MR": 'yale_results_preds_image_ModerateOrSevereMR_{}_min20.csv',
         "LVH": "yale_results_preds_image_HCM_LVDD_IVSd15_IntermediateAsFalse_{}_min20.csv",

         "SHD": "yale_results_preds_image_cmp_HCM_LVDD_ModtoSevVD_EF40_{}_min20.csv",
     }

all_preds1 = [
    "preds_image_HCM_LVDD_IVSd15_IntermediateAsFalse",
    "preds_image_ModerateOrSevereAS",
    "preds_image_ModerateOrSevereMR",
    "preds_image_Under40",
    'xgb_pred_cmp_HCM_LVDD_ModtoSevVD_EF40',
    'preds_male_sex_model_unfrozen_ep5'
]

columns_phewas = ['description', 'group', 'model',
                 'beta', 'SE', 'OR', 'p', 'n_total', 'n_cases', 'n_controls'
                 ]

# Results from Fisher exact
cohorts = ["YNHH", "Community", "NEMG", "UKB"]
num = 1
for cohort in cohorts:
    df = pd.read_csv(os.path.join(tabs_path, f"results_obs_exp_{cohort}.csv"))
    project.make_sup_table(save_path, f'Table {num}', string_dict[cohort], df)
    num+=1

# %%Results for PheWas
cohorts = ["YNHH", "Community", "NEMG"]
# Yale

def add_confidence_intervals(df):
    # Calculate 95% CI lower and upper bounds
    df['CI_lower'] = np.exp(df['beta'] - 1.96 * df['SE'])
    df['CI_upper'] = np.exp(df['beta'] + 1.96 * df['SE'])
    return df

# Yale
for cohort in cohorts:
    yale_list = []
    for var in project.all_preds_yale:
        df = pd.read_csv(os.path.join(project.tabs_path, f'yale_results_{var}_{cohort}_min20.csv'))
        df['model'] = project.preds_map_yale[var]
        df = df[columns_phewas]
        df = add_confidence_intervals(df)  # Add CI columns
        yale_list.append(df)
    yale_cox = pd.concat(yale_list)
    yale_cox.reset_index(drop=True, inplace=True)
    yale_cox = yale_cox.loc[:, ~yale_cox.columns.str.contains('Unnamed')]
    yale_string = f"{project.cohort_map_box[cohort]}: Results of the PheWAS for all AI-ECG models"
    project.make_sup_table(save_path, f'Table {num}', yale_string, yale_cox)
    num += 1

# UKB
UKB_list = []
for var in all_preds1:
    df = pd.read_csv(os.path.join(project.tabs_path, f'results_{var}_min20.csv'))
    df['model'] = project.preds_map_ukb[var]
    df = df[columns_phewas]
    df = add_confidence_intervals(df)  # Add CI columns
    UKB_list.append(df)
ukb_cox = pd.concat(UKB_list)
ukb_cox.reset_index(drop=True, inplace=True)
ukb_cox = ukb_cox.loc[:, ~ukb_cox.columns.str.contains('Unnamed')]
ukb_string = "UK Biobank: Results of the PheWAS for all AI-ECG models"
project.make_sup_table(save_path, f'Table {num}', ukb_string, ukb_cox)
num += 1

#%% Results for cox regression
cohorts = ["YNHH", "Community", "NEMG"]
# Yale
for cohort in cohorts:
    yale_list = []
    for var in project.all_preds_yale:
        df = pd.read_csv(os.path.join(project.tabs_path, f'yale_results_cox_noCMP_{var}_{cohort}.csv'))
        df['model'] = project.preds_map_yale[var]
        yale_list.append(df)
    yale_cox = pd.concat(yale_list)
    yale_cox.reset_index(drop=True, inplace=True)
    yale_cox = yale_cox.loc[:, ~yale_cox.columns.str.contains('Unnamed')]
    ukb_string = f"{project.cohort_map_box[cohort]}: Results of the multivariate cox regression for all AI-ECG models against cardiovascular phenoypes"
    project.make_sup_table(save_path, f'Table {num}', ukb_string, yale_cox)
    num += 1

# UKB
UKB_list = []
for var in project.all_preds:
    df =pd.read_csv(os.path.join(project.tabs_path, f'results_cox_noCMP_{var}.csv'))
    df['model'] = project.preds_map_ukb[var]
    UKB_list.append(df)
ukb_cox = pd.concat(UKB_list)
ukb_cox.reset_index(drop=True, inplace=True)
ukb_cox = ukb_cox.loc[:, ~ukb_cox.columns.str.contains('Unnamed')]
ukb_string = "UK Biobank: Results of the multivariate cox regression for all AI-ECG models against cardiovascular phenoypes"
project.make_sup_table(save_path, f'Table {num}', ukb_string, ukb_cox)
num += 1


print("====== DONE ======")
    
    
# %%
