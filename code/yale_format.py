"""
Script to prepare the yale data for PheWAS
"""


#  %%%
import pandas as pd
import numpy as np
import os
import project_constants as project
from tqdm import tqdm
import gc
print('start')

# %%
# Create dictionary to store counts
flow_diagram_counts = {}

print('Start processing')

# %%
# Load training cohort MRNs
excl_MRNs = []


SHD_train_path = os.path.join(project.yale_ecg_path, 
                                   "cards_data_feb_2024_filter_only_for_before_proc/train_ecg_echo_pairs_30days_15feb24_filter_only_for_before_proc.csv")
SHD_train = pd.read_csv(SHD_train_path, usecols=['MRN'])
excl_MRNs.extend(SHD_train['MRN'].tolist())

SHD_val_path = os.path.join(project.yale_ecg_path, 
                                   "cards_data_feb_2024_filter_only_for_before_proc/val_ecg_echo_single_pairs_30days_15feb24_filter_only_for_before_proc.csv")
SHD_val = pd.read_csv(SHD_val_path, usecols=['MRN'])
print(SHD_val.columns.to_list())

excl_MRNs.extend(SHD_val['MRN'].tolist())

excl_MRNs = list(set(excl_MRNs))
print(len(excl_MRNs))

# %%% paths
path_t2dm = '/home/pmc57/PheWas_AI_ECG/t2dm-jdat-data'
path_cmp = '/home/pmc57/PheWas_AI_ECG/cmp-jdat-data'
missing_cohort_path = os.path.join(path_t2dm, "ECG&Echo_Missing_JDAT/")
missing_update_path = os.path.join(path_cmp, "2023.03.28 ECG&Echo_MissingMRNs")

# %%
temp = pd.read_csv(os.path.join(missing_cohort_path, '2435227_ECG_Demographics.txt'),
                             sep='\t', on_bad_lines='skip', low_memory=False,nrows=10)
print(temp.columns.to_list())
# %% Loading and concatenating demographics
'PatientAge_ECGData'

# %%% Load ECG metadata
ecg = pd.read_csv(os.path.join(path_t2dm, "echo_ecg_sets/"
                'ecg_metadata_flagged_01jan2000_to_14jan2024.csv'), low_memory=False)

# 1️⃣ **Total ECGs & Patients Before Exclusion**
flow_diagram_counts['Total_ECGs_Before_Exclusion'] = len(ecg)
flow_diagram_counts['Total_Patients_Before_Exclusion'] = len(ecg['MRN'].unique())

print(ecg.groupby('Clean_InstitutionName')['MRN'].nunique())

# Remove training cohort from ECG dataset
ecg = ecg[~ecg['MRN'].isin(excl_MRNs)]

print(ecg.groupby('Clean_InstitutionName')['MRN'].nunique())

# 2️⃣ **Total ECGs & Patients After Excluding Training Cohort**
flow_diagram_counts['Total_ECGs_After_Exclusion_training'] = len(ecg)
flow_diagram_counts['Total_Patients_After_Exclusion_training'] = len(ecg['MRN'].unique())

# add predictions
preds = pd.read_csv(os.path.join(path_t2dm, "echo_ecg_sets/"
                'all_ecgs_with_image_preds_final.csv'), low_memory=False)

xgb = pd.read_csv(os.path.join(path_t2dm, "echo_ecg_sets/"
                'xgb_image_preds_all_ecgs.csv'), low_memory=False)

preds = preds.merge(xgb[['fileID', 'preds_image_cmp_HCM_LVDD_ModtoSevVD_EF40']], on = 'fileID', how = 'left')

preds_HCM = pd.read_csv(os.path.join(path_t2dm, "echo_ecg_sets/"
                'HCM_preds_AllYaleImages.csv'), low_memory=False)

preds = preds.merge(preds_HCM[['fileID', 'preds_image_HCM_NatCVR']], on = 'fileID', how = 'left')

preds_sex = pd.read_csv(os.path.join(path_t2dm, "echo_ecg_sets/"
                'predicted_male_sex_all_ecgs.csv'), low_memory=False)
preds = preds.merge(preds_sex[['fileID', 'preds_image_MaleSex']], on = 'fileID', how = 'left')

preds_cols = ['preds_image_cmp_HCM_LVDD_ModtoSevVD_EF40',
'preds_image_Under40',
'preds_image_HCM_LVDD_IVSd15_IntermediateAsFalse',
'preds_image_ModerateOrSevereValveDisease',
'preds_image_ModerateOrSevereMR',
'preds_image_ModerateOrSevereAR',
'preds_image_ModerateOrSevereAS', 
'preds_image_MaleSex', 
'preds_image_HCM_NatCVR']

# Filter after blanking period
# To pick the earliest encounter of a patient in the healthcare system:
hosp_enc_1 = project.load_data_in_chunks(os.path.join(missing_cohort_path, '2435227_ECG_Hospital_Enc.txt'), 
                                        columns = ['PAT_MRN_ID', 'HOSP_ADMSN_DATE'], 
                                        chunk_size = 10**6, sep = '\t')
hosp_enc_1 = pd.concat(hosp_enc_1)

hosp_enc_2 = project.load_data_in_chunks(os.path.join(missing_update_path,'Data-2024-03-12', '2435227_ECG_Hospital_Enc.txt'), 
                                        columns = ['PAT_MRN_ID', 'HOSP_ADMSN_DATE'],
                                        chunk_size = 10**6, sep = '\t')
hosp_enc_2 = pd.concat(hosp_enc_2)

hosp_enc_1.rename(columns={'HOSP_ADMSN_DATE': 'enc_date'}, inplace=True)
hosp_enc_2.rename(columns={'HOSP_ADMSN_DATE': 'enc_date'}, inplace=True)

enc_dates = pd.concat([hosp_enc_1, hosp_enc_2])
enc_dates['enc_date'] = pd.to_datetime(enc_dates['enc_date'])

earliest_enc = enc_dates.sort_values(by = 'enc_date', ascending = True).drop_duplicates(subset=['PAT_MRN_ID'], keep = 'first')
earliest_enc = earliest_enc.rename(columns={'enc_date': 'earliest_enc_date', 'PAT_MRN_ID':'MRN'})

latest_enc = enc_dates.sort_values(by = 'enc_date', ascending = False).drop_duplicates(subset=['PAT_MRN_ID'], keep = 'first')
latest_enc = latest_enc.rename(columns={'enc_date': 'latest_enc_date', 'PAT_MRN_ID':'MRN'})

# To add 1 year to this date:
earliest_enc['after_blanking_year'] = earliest_enc['earliest_enc_date'] + pd.DateOffset(years=1)

ecg = ecg.merge(earliest_enc, how = 'left', on = 'MRN')
ecg = ecg.merge(latest_enc, how = 'left', on = 'MRN')

# Keep the first ECG after after_blanking_year.
# Filter for ECGs after the after_blanking_year. Then sort by ECGDate and drop duplicates. 
ecg_after_blanking = ecg[ecg['ECGDate'] > ecg['after_blanking_year']].sort_values(by = 'ECGDate', ascending = True)

# Count before filtering out ECGs after last encounter
before_filter = len(ecg_after_blanking)

# Filter out ECGs that occur after latest encounter
ecg_after_blanking = ecg_after_blanking[ecg_after_blanking['ECGDate'] <= ecg_after_blanking['latest_enc_date']]

# Count how many were removed
removed_due_to_enc_date = before_filter - len(ecg_after_blanking)

# Update flow diagram counts
flow_diagram_counts[f'Total_ECGs_after_blanking'] = len(ecg_after_blanking)
flow_diagram_counts[f'Total_Patients_after_blanking'] = len(ecg_after_blanking['MRN'].unique())
flow_diagram_counts[f'Removed_ECGs_after_last_encounter'] = removed_due_to_enc_date

print(ecg.groupby('Clean_InstitutionName')['MRN'].nunique())

# %%
for institution in ['Westerly', 'L_and_M', 'Bridgeport', 'Greenwich', 'NEMG', 'YNHH', ]:
# for institution in ['NEMG']:
    ecg_institution = ecg_after_blanking[ecg_after_blanking['Clean_InstitutionName']==institution]

    ecg_ppl_mrn_list = ecg_institution['MRN'].unique().tolist()
    ecg_institution['PAT_MRN_ID'] = ecg_institution['MRN'].copy()

    flow_diagram_counts[f'Total_ECGs_{institution}'] = len(ecg_institution)
    flow_diagram_counts[f'Total_Patients_{institution}'] = len(ecg_institution['MRN'].unique())
    
    study_pop = ecg_institution[['MRN', 'PAT_MRN_ID', 'ECGDate', 'after_blanking_year', 'fileID',  'fails_to_load_25jan2024', 'YearMonth', 'Year', 'Clean_InstitutionName', 'external_site_ecg', 'earliest_enc_date', 'latest_enc_date' ]]
    study_pop = study_pop.merge(demo, on ='PAT_MRN_ID', how = 'left')

    study_pop['BIRTH_DATE'] = pd.to_datetime(study_pop['BIRTH_DATE'])
    study_pop['ECGDate'] = pd.to_datetime(study_pop['ECGDate'])
    study_pop['Age_at_ECG'] = (study_pop['ECGDate'] - study_pop['BIRTH_DATE']).dt.days // 365.25

    study_pop = study_pop.merge(preds[['fileID'] + preds_cols], on = 'fileID', how = 'left')
    study_pop.dropna(subset = preds_cols, inplace=True)
    
     # Make sure only patients from that institution
    study_pop = study_pop[study_pop['Clean_InstitutionName'] == institution]

    # 
    ecg_ppl_mrn_list = study_pop[study_pop['PAT_MRN_ID'].isin(ecg_ppl_mrn_list)]['PAT_MRN_ID'].to_list()
    ecg_ppl_mrn_list = list(set(ecg_ppl_mrn_list))

    # Load required paths and constants
    chunk_size = 40000  # Set chunk size
    n_patients = len(ecg_ppl_mrn_list)  # Total number of unique MRNs
    print(f"Total patients {institution}: {n_patients}")

    # Initialize lists to store intermediate results
    all_earliest = []
    all_latest = []
    all_random = []
    all_list = []

    dx_df_cols = ['PAT_ID', 'PAT_MRN_ID', 'DX_SOURCE','DX_DATE', 'CURRENT_ICD10_LIST']

    missing_dx_files = [
            os.path.join(missing_cohort_path, f'2435227_ECG_Dx_{i}.txt') for i in range(1, 5)
        ]

    # Load missing update data
    missing_update_dx_files = [
            os.path.join(missing_update_path, f'2435227_ECG_Dx_{i}.txt') for i in range(1, 5)
        ]

    for start_idx in tqdm(range(0, n_patients, chunk_size), desc="Processing Chunks"):
        end_idx = min(start_idx + chunk_size, n_patients)
        current_chunk = ecg_ppl_mrn_list[start_idx:end_idx]

        # 
        filtered_dx_chunks = []

        for file in tqdm(missing_dx_files, desc="Processing Files", unit="file"):
            filtered_chunk = project.load_filtered_data(file, chunk_size=10**6, columns=dx_df_cols, sep='\t', 
                                                        incl_MRNs=current_chunk)
            filtered_dx_chunks.append(filtered_chunk)

        missing_dx = pd.concat(filtered_dx_chunks)
        del filtered_dx_chunks
        print(len(missing_dx))

        filtered_update_dx_chunks = []

        for file in tqdm(missing_update_dx_files, desc="Processing Files", unit="file"):
            filtered_chunk = project.load_filtered_data(
                file, 
                chunk_size=10**6, 
                columns=dx_df_cols, 
                sep='\t', 
                incl_MRNs=current_chunk
            )
            filtered_update_dx_chunks.append(filtered_chunk)
        missing_update_dx = pd.concat(filtered_update_dx_chunks)
        del filtered_update_dx_chunks

        dx = pd.concat([missing_dx, missing_update_dx])
        dx.dropna(subset=['DX_DATE', 'CURRENT_ICD10_LIST'], inplace=True)
        del missing_dx
        del missing_update_dx
        
        #Make long format for PheWAS package
        dx.dropna()
        dx = dx.sort_values(by=['PAT_MRN_ID', 'DX_DATE'])
        dx.assign(CURRENT_ICD10_LIST=dx['CURRENT_ICD10_LIST'].str.split(', ')).explode('CURRENT_ICD10_LIST')

        # Make sure only first instance appears
        dx = dx.drop_duplicates(subset=['PAT_MRN_ID', 'CURRENT_ICD10_LIST'], keep='first')

        # Add a sequential index for each ICD code within a patient
        dx['ICD_INDEX'] = dx.groupby('PAT_MRN_ID').cumcount()

        # Pivot to wide format for ICD codes
        dx_wide = dx.pivot(index='PAT_MRN_ID', columns='ICD_INDEX', values='CURRENT_ICD10_LIST')

        # Pivot to wide format for dates
        dx_date_wide = dx.pivot(index='PAT_MRN_ID', columns='ICD_INDEX', values='DX_DATE')

        # Flatten the MultiIndex columns for better readability
        dx_wide.columns = [f'ICD_code_{col}' for col in dx_wide.columns]
        dx_date_wide.columns = [f'ICD_date_{col}' for col in dx_date_wide.columns]

        # Combine the two DataFrames
        dx_long_format = pd.concat([dx_wide, dx_date_wide], axis=1).reset_index()

        # Modify and save the long format
        long_format = study_pop.merge(dx_long_format, on = 'PAT_MRN_ID', how = 'inner')
        del dx_long_format

        # 
        # Ensure 'ECG date' is in datetime format
        long_format['ECGDate'] = pd.to_datetime(long_format['ECGDate'])
        # Ensure ECG date is in datetime format
        long_format['ECGDate'] = pd.to_datetime(long_format['ECGDate'])
        # Sort by PAT_MRN_ID and ECG date to ensure proper ordering
        long_format = long_format.sort_values(by=['PAT_MRN_ID', 'ECGDate'], ascending=[True, True])  # Ascending order for earliest ECG
        # Deduplicate to keep the earliest ECG (first by default in sorted order)
        earliest = long_format.drop_duplicates(subset=['PAT_MRN_ID'], keep='first').reset_index(drop=True)

        latest = long_format.drop_duplicates(subset=['PAT_MRN_ID'], keep='last').reset_index(drop=True)

        # Deduplicate to keep a random ECG for each patient
        random = long_format.groupby('PAT_MRN_ID').apply(lambda x: x.sample(n=1, random_state=42)).reset_index(drop=True)
        total = len(random['PAT_MRN_ID'])
        unique = len(random['PAT_MRN_ID'].unique())
        print(f'total0 = {total}, unique = {unique}')
        
        # Append to result lists
        all_earliest.append(earliest)
        all_latest.append(latest)
        all_random.append(random)
        all_list.append(long_format)

        # Free memory
        del earliest, latest, random, long_format
        gc.collect()

    long_format_earliest = pd.concat(all_earliest, ignore_index=True)
    long_format_latest = pd.concat(all_latest, ignore_index=True)
    long_format_random = pd.concat(all_random, ignore_index=True)
    all_data = pd.concat(all_list, ignore_index=True)

    long_format_earliest.to_csv(os.path.join(project.tabs_path, f"Yale_ICD_long_format_earliest_cox_{institution}.tsv"), sep='\t')
    long_format_random.to_csv(os.path.join(project.tabs_path, f"Yale_ICD_long_format_random_cox_{institution}.tsv"), sep='\t')
    all_data.to_csv(os.path.join(project.tabs_path, f"Yale_ICD_long_format_all_cox_{institution}.tsv"), sep='\t')

    long_format_earliest_cs = project.update_icd_codes_and_dates(long_format_earliest, 'ECGDate', 
                                icd_pattern='ICD_code_', 
                                date_pattern='ICD_date_')

    long_format_earliest_cs.to_csv(os.path.join(project.tabs_path, f"Yale_ICD_long_format_earliest_{institution}.tsv"), sep='\t')


    del long_format_earliest

    long_format_latest = project.update_icd_codes_and_dates(long_format_latest, 'ECGDate', 
                                icd_pattern='ICD_code_', 
                                date_pattern='ICD_date_')
    long_format_latest.to_csv(os.path.join(project.tabs_path, f"Yale_ICD_long_format_latest_{institution}.tsv"), sep='\t')
    del long_format_latest

    #

    long_format_random = project.update_icd_codes_and_dates(long_format_random, 'ECGDate', 
                                icd_pattern='ICD_code_', 
                                date_pattern='ICD_date_')


    # Only take the last available ECG, remove ICD codes after ECG date
    long_format_random.to_csv(os.path.join(project.tabs_path, f"Yale_ICD_long_format_random_{institution}.tsv"), sep='\t')

    del long_format_random

# Save counts to CSV for tracking
flow_counts_df = pd.DataFrame(list(flow_diagram_counts.items()), columns=['Step', 'Count'])
flow_counts_df.to_csv(os.path.join(project.tabs_path, "flow_diagram_counts.csv"), index=False)

# %%
