"""
This script fits cox models in the UK Biobank
"""


# %%
import numpy as np 
import pandas as pd
import os
import project_constants as project 
from constants import (

    baseline_table,
)

from tqdm import tqdm
from datetime import datetime
import time
from lifelines import CoxPHFitter
import ast
from sklearn.preprocessing import StandardScaler
from lifelines.statistics import proportional_hazard_test
import matplotlib.pyplot as plt

print('start')
constructor = project.PhenotypeConstructor()
# %%%
# Start the timer
start_time = time.time()

# %%
# load data

load_path = os.path.join(project.tabs_path, "UKB_with_Arrays_temp.csv")

data = pd.read_csv(load_path)

print(data.columns.to_list())

path_t2dm = '/home/pmc57/PheWas_AI_ECG/t2dm-jdat-data'

# %%%
# Save the data to a file
ecg_data = pd.read_csv('/home/pmc57/PheWas_AI_ECG/data_2/data_LD/ukb_image_preds_final_WithXGB.csv')
ecg_data = ecg_data[ecg_data['ecg_instance'] == 2]
HCM_data = pd.read_csv('/home/pmc57/PheWas_AI_ECG/data_2/bb_HCM_preds.csv')

# sex_data = pd.read_csv('/home/pmc57/PheWas_AI_ECG/data_2/data_LD/ukb_male_sex.csv')
sex_data = pd.read_csv(os.path.join(path_t2dm, "echo_ecg_sets/", "predicted_male_sex_ukb_ecgs_feb6.csv"))
data = pd.merge(ecg_data, data, on='eid', how = 'left')
data = pd.merge(HCM_data[['HCM_Pred', 'PID']], data, on='PID', how = 'inner')
data = pd.merge(sex_data[['preds_image_MaleSex', 'PID']], data, on='PID', how = 'inner')
print(data)
data[data['ecg_instance'] == 2]

# %%
phecode_map_icd10 = pd.read_csv(os.path.join(project.tabs_path, "phecode_map_icd10.csv"))
phecode_sub = pd.read_csv(os.path.join(project.tabs_path, "yale_results_preds_image_ModerateOrSevereAS_min20.csv"))

phecode_to_description = phecode_sub.set_index(['phenotype'])['description'].to_dict()
phecode_map_icd10 = phecode_map_icd10[phecode_map_icd10['phecode'].isin(phecode_sub['phenotype'])]

# %%%
icd_to_phecode_dict = phecode_map_icd10.set_index(['code'])['phecode'].to_dict()

data['Diagnoses_ICD10_array_tmp'] = data['Diagnoses_ICD10_array'].apply(project.map_icd_to_phecode, args=(icd_to_phecode_dict,))

print(data['Diagnoses_ICD10_array_tmp'])
# %%

add_columns = project.include_columns + project.all_preds
print(data.columns.to_list)
data_with_phecode_dates = project.add_phecode_date_columns(data, 
                                 phecode_to_description, additional_columns=add_columns)

# %%%
independent_var = "preds_image_HCM_LVDD_IVSd15_IntermediateAsFalse"

description_columns =  phecode_sub['description'].to_list()
print(len(description_columns))

# %%
no_CMP = data_with_phecode_dates.copy()

no_CMP['date_of_attending_assessment_centre_f53_2_0'] = \
    pd.to_datetime(no_CMP['date_of_attending_assessment_centre_f53_2_0'], errors='coerce')
    
no_CMP['date_lost_to_followup_f191_0_0'] = \
    pd.to_datetime(no_CMP['date_lost_to_followup_f191_0_0'], errors='coerce')

no_CMP = \
    no_CMP[(no_CMP["date_of_attending_assessment_centre_f53_2_0"] < \
        no_CMP['date_lost_to_followup_f191_0_0']) | no_CMP['date_lost_to_followup_f191_0_0'].isna()]
    
count = len(no_CMP)
print(count)
for excl in project.sig_list_curated_sub:
    if excl in no_CMP.columns:  # Only proceed if column exists
        no_CMP[excl] = pd.to_datetime(no_CMP[excl], errors='coerce')  # Ensure dates are properly converted
        no_CMP = no_CMP[(no_CMP[excl].isnull()) | (no_CMP[excl] > no_CMP['date_of_attending_assessment_centre_f53_2_0'])]  # Remove rows where ICD code is before or onECGDate
        
        print(f'{excl} removed {count - len(no_CMP)}')  # Print how many rows were removed for this ICD code
        count = len(no_CMP)  # Update count after filtering
        print(no_CMP[excl].notna().sum())  # Print remaining non-null values in this column
    else:
        print(f"Skipping {excl} - column not found in DataFrame")

# Initialize the scaler
scaler = StandardScaler()

print(len(no_CMP))
        
no_CMP[project.sig_list_curated_sub + 
           project.include_columns].to_csv(os.path.join(project.tabs_path, 
                                                             f'no_CMP_UKB.csv'))

# %%%
# Standardization in the cox_results function!!
# Prepare UKB event columns
descriptions = [desc for desc in project.sig_list_curated_sub if desc in no_CMP.columns]

for var in project.all_preds:
    print(f"Running Cox for predictor: {var}")
    
    if independent_var == 'preds_male_sex_model_unfrozen_ep5':
        adjust_vars = ['age_when_attended_assessment_centre_f21003_2_0']
        
    else: 
        adjust_vars=['age_when_attended_assessment_centre_f21003_2_0', 'sex_f31_0_0']
            
    
    cox_results = project.cox_regression(
        df=no_CMP,
        description_columns=descriptions,
        independent_var=var,
        adjust_vars=adjust_vars,
        baseline_col='date_of_attending_assessment_centre_f53_2_0',
        death_col='date_of_death_f40000_0_0',
        censoring_date='2023-12-31',
        max_period_days=365 * 5,
        stand_log=True
    )

    # Output
    print(cox_results)
    output_path = os.path.join(project.tabs_path, f'results_cox_noCMP_{var}.csv')
    cox_results.to_csv(output_path, index=False)
