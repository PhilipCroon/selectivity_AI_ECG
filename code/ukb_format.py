"""
Script to prepare the UKB data for PheWAS
"""


# %%%
import numpy as np 
import pandas as pd
import os
import project_constants as project 
from tqdm import tqdm
from datetime import datetime
from constants import (

    baseline_table,
)

print("start")
path_t2dm = '/home/pmc57/PheWas_AI_ECG/t2dm-jdat-data'
max_date = datetime.strptime("2020-07-08", "%Y-%m-%d") # After we don't have followup. 
raw_data_path = os.path.join(project.data_path, "ukb47034.csv")

#%%

    # Function to handle repetitive self-reported condition conversion
def convert_self_reported_conditions(data, base_name, index_range=4):
    """Convert self-reported condition columns to lists for each index."""
    for i in range(index_range):
        cols = [col for col in data.columns if f'{base_name}_{i}' in col]
        data[f"{base_name}_SR_{i}_0"] = data[cols].apply(lambda row: [int(x) for x in row if pd.notna(x)], axis=1).tolist()

# Initialize constructor
constructor = project.PhenotypeConstructor()

df = pd.read_csv(raw_data_path, nrows=0)
col_list = df.columns.to_list()
print(project.UKB_colmap)
field_keys_str = [str(key) for key in project.UKB_colmap.keys()]

# Create a list of columns that contain the FieldID as a substring
matching_cols = [col for col in col_list if any(col.startswith(f"{key}-") for key in field_keys_str)]

# Load the data
raw_data = project.load_data_in_chunks(raw_data_path, matching_cols + ['eid'])

# %%%
# making a sub with ECG for faster coding

# # save dataframe
df = pd.read_csv(os.path.join(project.tabs_path, f'Clean_diagnoses_full.tsv'),
                             sep='\t')
icd_codes = pd.read_csv(os.path.join(project.tabs_path, "UKB_with_Arrays_temp.csv"))

# %%
# Get list of column names
column_list = icd_codes.columns.tolist()
# Filter columns that start with the desired prefixes
selected_columns = [col for col in column_list if col.startswith("date_of_first_in-patient_diagnosis_icd10_f")]

print(icd_codes[selected_columns])

# %%
# Get list of column names
column_list = icd_codes.columns.tolist()
# Filter columns that start with the desired prefixes
selected_columns = [col for col in column_list if col.startswith("diagnoses_icd10_f") or col.startswith("date_of_first_in-patient_diagnosis_icd10_f")]

icd_codes = icd_codes[selected_columns+['eid']]
print(icd_codes.head())

# %%
                  
ecg_data = pd.read_csv('/home/pmc57/PheWas_AI_ECG/data_2/data_LD/ukb_image_preds_final_WithXGB.csv')
sex_data = pd.read_csv(os.path.join(path_t2dm, "echo_ecg_sets/", "predicted_male_sex_ukb_ecgs_feb6.csv"))
sex_data = sex_data.drop_duplicates(subset="PID")
# Only instance 2
ecg_data = ecg_data[ecg_data['ecg_instance']==2]
ecg_data = ecg_data[~ecg_data['PID'].duplicated(keep=False)]

HCM_data = pd.read_csv('/home/pmc57/PheWas_AI_ECG/data_2/bb_HCM_preds.csv') 
HCM_data['ecg_instance'] = HCM_data['filename'].str.split('_').str[2]
HCM_data = HCM_data[HCM_data['ecg_instance']=="2"]
HCM_data = HCM_data[~HCM_data['PID'].duplicated(keep=False)]
HCM_data
print(len(HCM_data))

merged_df = pd.merge(ecg_data, df, on='eid', how = 'left')
print(1, len(merged_df))
merged_df = pd.merge(merged_df, icd_codes, on='eid', how = 'left')
print(2, len(merged_df))
merged_df = pd.merge(merged_df, sex_data[['preds_image_MaleSex', 'PID']], on='PID', how = 'left')
print(3, len(merged_df))
merged_df['preds_male_sex_model_unfrozen_ep5'] = merged_df['preds_image_MaleSex'] 
merged_df = pd.merge(HCM_data[['HCM_Pred', 'PID']], merged_df, on='PID', how = 'left')
print(3, len(merged_df))

merged_df['HCM_b'] = (merged_df['HCM_Pred'] >= 0.15).astype(int)
merged_df['LVF_b'] = (merged_df['preds_image_Under40'] >= 0.1).astype(int)
merged_df['VHD_b'] = (merged_df["preds_image_ModerateOrSevereValveDisease"] >= 0.17570518).astype(int)

# %%%

# OPtional add old icd cols
save_path = os.path.join(project.tabs_path, "UKB_with_Arrays_ECG.csv")

# Save the data to a file
merged_df.to_csv(save_path)

print(f'data saved to {save_path}')

# %%%
save_path = os.path.join(project.tabs_path, "UKB_with_Arrays_ECG.csv")
merged_df = pd.read_csv(save_path)

# For pre-PheWas
pre_pws = project.update_icd_codes_and_dates_ukb(merged_df, 'date_of_attending_assessment_centre_f53_2_0')
print("ICD codes and dates updated")

pre_pws.to_csv(os.path.join(project.tabs_path, "UKB_pre_PheWas_ecg.csv"))
print(1, len(pre_pws))
print('Pre pws data saved')
print(pre_pws['preds_male_sex_model_unfrozen_ep5'])
# %%
print(len(pre_pws))
# %%
