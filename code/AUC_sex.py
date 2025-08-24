"""
Script: evaluate_sex_prediction_model.py

Purpose:
    This script evaluates the performance of a deep learning model for sex prediction from ECGs 
    across multiple cohorts (Yale New Haven Hospitals, Community Hospitals, Outpatient Clinics, 
    and UK Biobank).

Main Steps:
    1. Load prediction and demographic data for Yale and UK Biobank.
    2. Define institution-based cohorts and harmonize sex labels into binary format.
    3. Compute ROC curves and AUC values for each cohort.
    4. Perform bootstrapping (100 iterations) to estimate 95% confidence intervals for the AUCs.
    5. Save summary results as a CSV file and plot ROC curves for comparison.

Outputs:
    - auc_results.csv : AUC values with 95% CI for each cohort.
    - ROC curve sex.pdf : ROC curves for all cohorts in a single figure.
"""

# %%
import numpy as np
import pandas as pd
import os
import project_constants as project
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.utils import resample

# Define paths
ukb_path = os.path.join(project.tabs_path, "UKB_pre_PheWas_ecg.csv")
yale_path = os.path.join(project.tabs_path, "Yale_ICD_long_format_random_cox_all_hospitals.tsv")

# Load data
ukb = pd.read_csv(ukb_path)
yale = pd.read_csv(yale_path, sep='\t')

# **Define cohorts from Yale**
cohorts = {
    "YNHH": yale[yale['institution'] == "YNHH"],
    "Community": yale[yale['institution'].isin(['Westerly', 'L_and_M', 'Bridgeport', 'Greenwich'])],
    "NEMG": yale[yale['institution'] == "NEMG"],
    "UKB": ukb
}
# Define the color mapping
cohort_colors = {
    "YNHH": "#1f3b73",  # Dark blue for Yale New Haven Hospitals
    "UKB": "#F09EA7",   # Light red/pink for UK Biobank
    "Community": "#F6CA94",  # Warm orange for Community hospitals
    "NEMG": "#B0D2EB"   # Soft blue for Outpatient clinics
}

# **Cohort name mapping**
cohort_map_box = {
    "YNHH": "Yale New Haven Hospitals",
    "UKB": "UK Biobank",
    "NEMG": "Outpatient Clinics",
    "Community": "Community Hospitals"
}

# **Map sex labels**
for cohort_name, df in cohorts.items():
    if cohort_name == "UKB":
        df["sex_binary"] = df["sex_f31_0_0"]  # Female = 1, Male = 0
    else:
        df["sex_binary"] = df["SEX"].map({"Female": 0, "Male": 1})  # Female = 1, Male = 0

# **Perform bootstrap AUC calculation & Plot ROC Curves**
bootstrap_iterations = 100
auc_results = {}
plt.figure(figsize=(3, 3))

for cohort_name, df in cohorts.items():
    if cohort_name == "UKB":
        pred_col = 'preds_male_sex_model_unfrozen_ep5'
    else:
        pred_col = 'preds_image_MaleSex'
    print(f"Computing AUC for {cohort_name}...")

    preds = df[pred_col].dropna()  # Drop NaN values
    labels = df.loc[preds.index, "sex_binary"]  # Align labels with predictions
    
    boot_auc_scores = []
    boot_fprs, boot_tprs = [], []
    
    for _ in range(bootstrap_iterations):
        sample_preds, sample_labels = resample(preds, labels, stratify=labels)  # Bootstrap sample
        auc_value = roc_auc_score(sample_labels, sample_preds)
        boot_auc_scores.append(auc_value)
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(sample_labels, sample_preds)
        boot_fprs.append(fpr)
        boot_tprs.append(tpr)
    
    # Compute mean ROC curve
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(boot_fprs, boot_tprs)], axis=0)
    
    # Store AUC results
    auc_results[cohort_map_box[cohort_name]] = {
        "mean_auc": np.mean(boot_auc_scores),
        "lower_ci": np.percentile(boot_auc_scores, 2.5),  # 2.5th percentile
        "upper_ci": np.percentile(boot_auc_scores, 97.5)  # 97.5th percentile
    }

# Convert the dictionary to a DataFrame
auc_df = pd.DataFrame(auc_results).T.reset_index()
auc_df.columns = ["Cohort", "Mean AUC", "Lower CI", "Upper CI"]
print(auc_df)
# Define the save path
save_path = os.path.join(project.tabs_path, "auc_results.csv")

# Save as CSV
auc_df.to_csv(save_path, index=False)

print(f"AUC results saved to {save_path}")

# %%

# Initialize figure
plt.figure(figsize=(6, 6))

# Loop through cohorts and plot AUC curves
for cohort_name, df in cohorts.items():
    # Determine prediction column
    pred_col = 'preds_male_sex_model_unfrozen_ep5' if cohort_name == "UKB" else 'preds_image_MaleSex'
    
    preds = df[pred_col].dropna()  # Drop NaN values
    labels = df.loc[preds.index, "sex_binary"]  # Align labels with predictions

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve with the assigned color
    plt.plot(fpr, tpr, color=cohort_colors[cohort_name], lw=2, 
             label=f"{cohort_map_box[cohort_name]} (AUC = {roc_auc:.2f})")

# Plot the diagonal reference line
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)

# Formatting
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating curves for Sex Prediction Model")
plt.legend(loc="lower right", fontsize=10, frameon=False)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid(False)  # Remove grid

# Show plot
plt.savefig(os.path.join(project.tabs_path, "ROC curve sex.pdf"))
plt.show()

# %%
