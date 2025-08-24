"""
Cox regression in Yale
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
import gc
from lifelines.statistics import proportional_hazard_test

print('start')
constructor = project.PhenotypeConstructor()
# %%%
# Start the timer
start_time = time.time()

counts_dict = {}

def map_dates_to_phecodes(row, phecode_to_description):
    """
    Maps ICD codes to their corresponding phecodes and assigns the earliest date.
    If an ICD code is not found in the mapping dictionary, it is skipped.
    """
    # Initialize dictionary with NaT (missing dates)
    dates_dict = {description: pd.NaT for description in phecode_to_description.values()}

    # Extract phecodes and dates
    phecodes_array = row['Diagnoses_ICD10_array_tmp']
    dates_array = row['Date_of_first_in_patient_diagnosis_ICD10_array']

    # Ensure both arrays are converted properly
    try:
        phecodes_array = ast.literal_eval(phecodes_array) if isinstance(phecodes_array, str) else list(phecodes_array)
    except:
        phecodes_array = []
    
    try:
        dates_array = ast.literal_eval(dates_array) if isinstance(dates_array, str) else list(dates_array)
    except:
        dates_array = []

    # **Filter out None values from phecodes_array and align dates_array accordingly**
    filtered_pairs = [(phecode, date) for phecode, date in zip(phecodes_array, dates_array) if phecode is not None]

    # If no valid phecodes remain, return empty dictionary
    if not filtered_pairs:
        return dates_dict

    # Map phecodes to descriptions while maintaining the earliest date
    for phecode, date in filtered_pairs:
        description = phecode_to_description.get(phecode)
        if description:  # Only keep known phecodes
            date = pd.to_datetime(date, errors='coerce')
            if pd.isna(dates_dict[description]) or date > dates_dict[description]:  # Keep the earliest date
                dates_dict[description] = date

    return dates_dict

def add_phecode_date_columns(data, phecode_to_description, additional_columns):
    """
    Applies mapping function row by row, ensuring only known phecodes are included.
    """
    # Apply the mapping function with tqdm progress bar
    phecode_dates = data.progress_apply(lambda row: map_dates_to_phecodes(row, phecode_to_description), axis=1)
    
    # Convert the result to a DataFrame with the corresponding column names
    phecode_dates_df = pd.DataFrame(phecode_dates.tolist(), index=data.index)
    
    # Select the additional columns from the original data
    additional_columns_df = data[additional_columns]
    
    # Create a new DataFrame with the phecode columns and the additional columns
    new_dataframe = pd.concat([phecode_dates_df, additional_columns_df], axis=1)
    
    return new_dataframe

# %%
# load data
# Make a community hospital df
comm_list = []
for institution in ['YNHH', 'Westerly', 'L_and_M', 'Bridgeport', 'Greenwich', 'NEMG']:
    path = os.path.join(project.tabs_path, f"Yale_ICD_long_format_random_cox_{institution}.tsv")
    df = pd.read_csv(path, sep='\t')
    df['institution'] = institution
    comm_list.append(df)

# %%
comm_df = pd.concat(comm_list)
comm_df = comm_df.drop_duplicates(subset=['PAT_MRN_ID'], keep='first')

# comm_df.to_csv(os.path.join(project.tabs_path, f"Yale_ICD_long_format_random_cox_all_hospitals.tsv"), sep='\t')

# %%%
for institution in ['YNHH','Community', 'NEMG']: 
    no_CMP = pd.read_csv(os.path.join(project.tabs_path, 
                                                    f'no_CMP_{institution}.csv'))
    print(len(no_CMP))
    # remove individuals with dead date before ECG (only a few..)
    no_CMP = no_CMP[(no_CMP['DEATH_DATE'].isna()) | (no_CMP['DEATH_DATE'] > no_CMP['ECGDate'])]
    print(len(no_CMP))
    # Determine censoring date from data
    censoring_date = no_CMP['latest_enc_date'].max()

    # Optionally enforce uniform censoring date across dataset
    no_CMP['latest_enc_date'] = censoring_date  

    # Convert sex to binary format
    no_CMP['SEX'] = no_CMP['SEX'].map({'Female': 1, 'Male': 0})
    no_CMP['ECGDate'] = pd.to_datetime(no_CMP['ECGDate'], errors='coerce')
    no_CMP['BIRTH_DATE'] = pd.to_datetime(no_CMP['BIRTH_DATE'], errors='coerce')
    no_CMP['Age_at_ECG'] = (no_CMP['ECGDate'] - no_CMP['BIRTH_DATE']).dt.days / 365.25

    # Ensure all the relevant event columns exist in the dataset
    descriptions = [description for description in project.sig_list_curated_sub if description in no_CMP.columns]

# Loop over predictor variables
    for var in project.all_preds_yale:
        print(var)

        # Choose adjustment variables
        if var == 'preds_male_sex_model_unfrozen_ep5':
            adjust = ['Age_at_ECG']
        else:
            adjust = ['Age_at_ECG', 'SEX']

        # Run Cox regression
        cox_results = project.cox_regression(
            df=no_CMP,
            description_columns=descriptions,
            independent_var=var,
            cohort=institution, 
            adjust_vars=adjust,
            baseline_col='ECGDate',
            death_col='DEATH_DATE',
            censoring_date=censoring_date,
            max_period_days=365*5,
            stand_log=True
        )

        # Output results
        print(cox_results)
        print('done')
     
        cox_results.to_csv(os.path.join(project.tabs_path, f'yale_results_cox_noCMP_{var}_{institution}.csv'))
        
# Convert dictionary to DataFrame
# df = pd.DataFrame(list(counts_dict.items()), columns=['Institution', 'Count'])
# Save to CSV
# df.to_csv(os.path.join(project.tabs_path, "counts_cox.csv"), index=False)
# %%
