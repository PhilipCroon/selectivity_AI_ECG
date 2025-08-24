rm(list = ls())
# Load necessary libraries
library(dplyr)
library(PheWAS)

# Load the UK Biobank data
ukb <- read.csv("/Users/philipcroon/PycharmProjects/Yale/DL_ECG_PheWAS/data/UKB_pre_PheWas_ecg.csv")
ukb <- ukb %>% filter(ecg_instance == 2)

# Add exp outcomes
exp <- read.delim("/Users/philipcroon/PycharmProjects/Yale/DL_ECG_PheWAS/data/exp_preds_ukb.csv",
                  header = TRUE, sep = ",", stringsAsFactors = FALSE)
exp <- exp %>% distinct(PID, .keep_all = TRUE)

# Merge on PAT_MRN_ID (inner join by default)
ukb <- merge(ukb, exp, by = "PID", how = 'left')

set.seed(123)  # (optional) Set seed for reproducibility
ukb$Random <- runif(nrow(ukb))

cat("Number of patients", nrow(ukb), "\n")

# Define a list of AI models you want to test
models <- c(
            "preds_image_HCM_LVDD_IVSd15_IntermediateAsFalse",
            "preds_image_ModerateOrSevereAS",
            "preds_image_ModerateOrSevereMR",
            "preds_image_Under40",
            "preds_image_ModerateOrSevereValveDisease",
              "xgb_pred_cmp_HCM_LVDD_ModtoSevVD_EF40",
              "HCM_Pred",
           "preds_male_sex_model_unfrozen_ep5")

# models <- c('Random',
#   'preds_Dermatophytosis',
#   'preds_image_EvenMonth',
#   'preds_Bitten_by_dog',
#   'preds_Fractures',
#   'preds_Headache',
#   'preds_Transport_accident',
#   'preds_Viral_respiratory_infection'
# )
# 
# models <- c('Random'
# )


# First set of columns: eid and ICD-10 diagnosis codes
desired_columns_1 <- c("eid", grep("^diagnoses_icd10_f41270_0", names(ukb), value = TRUE))

# Second column to add: age information
desired_columns_2 <- "age_when_attended_assessment_centre_f21003_2_0"

# Third set of columns: all possible AI model columns and sex information
desired_columns_3 <- c("eid", 
                       "preds_image_HCM_LVDD_IVSd15_IntermediateAsFalse", 
                       "preds_image_ModerateOrSevereAS", 
                       "preds_image_ModerateOrSevereMR", 
                       "preds_image_Under40", 
                       "preds_image_ModerateOrSevereValveDisease", 
                       "xgb_pred_cmp_HCM_LVDD_ModtoSevVD_EF40",
                       'preds_male_sex_model_unfrozen_ep5',
                       "HCM_Pred",
                       "sex_f31_0_0", 
                       'preds_Dermatophytosis', 
                       'preds_image_EvenMonth', 
                       'preds_Bitten_by_dog', 
                       'preds_Fractures', 
                       'preds_Headache', 
                       'preds_Transport_accident', 
                       'preds_Viral_respiratory_infection', 
                       'Random')

# Combine all the columns into a single vector and remove duplicates
desired_columns <- unique(c(desired_columns_1, desired_columns_2, desired_columns_3))
merged <- ukb[, desired_columns]

# Apply logit transformation and standardization for each model in the vector
for (model in models) {
  # Logit transformation
  merged[[model]] <- log(merged[[model]] / (1 - merged[[model]]))
  
  # Standardization (subtract mean and divide by standard deviation)
  merged[[model]] <- scale(merged[[model]], center = TRUE, scale = TRUE)
}

# Calculate percentage of males
male_percentage <- sum(merged$sex_f31_0_0 == 1, na.rm = TRUE) / sum(!is.na(merged$sex_f31_0_0)) * 100

# Calculate mean and standard deviation of age
mean_age <- mean(merged$age_when_attended_assessment_centre_f21003_2_0, na.rm = TRUE)
sd_age <- sd(merged$age_when_attended_assessment_centre_f21003_2_0, na.rm = TRUE)

# Print demographic results
cat("Mean age: ", mean_age, "\n")
cat("Standard deviation of age: ", sd_age, "\n")
cat("Percentage of male: ", male_percentage, "%\n")

# Load phecode map file (assuming it's already loaded in your environment)
phecode_map_icd10$code <- gsub("\\.", "", phecode_map_icd10$code)

# Initialize phenotype dataframe
phenotypes <- data.frame(id = merged$eid)
unique_phecodes <- unique(phecode_map_icd10$phecode)
for (phecode in unique_phecodes) {
  phenotypes[[phecode]] <- FALSE
}

update_phenotypes <- function(merged, phecode_map_icd10) {
  # Clean ICD codes in the phecode map
  phecode_map_icd10$code <- gsub("\\.", "", phecode_map_icd10$code)
  
  # Identify ICD-10 diagnosis columns
  diagnosis_cols <- grep("^diagnoses_icd10_f41270_0", names(merged), value = TRUE)
  
  # Clean ICD codes in the merged data
  for (col in diagnosis_cols) {
    merged[[col]] <- gsub("\\.", "", merged[[col]])
  }
  
  # Convert wide to long format
  icd_long <- merged %>%
    select(eid, all_of(diagnosis_cols)) %>%
    tidyr::pivot_longer(
      cols = -eid,
      names_to = "icd_col",
      values_to = "icd_code"
    ) %>%
    filter(!is.na(icd_code), icd_code != "")
  
  # Join with phecode map
  icd_phecode <- icd_long %>%
    left_join(phecode_map_icd10, by = c("icd_code" = "code"), relationship = "many-to-many") %>%
    filter(!is.na(phecode), phecode != "")
  
  # Create phenotype matrix
  phenotypes <- icd_phecode %>%
    distinct(eid, phecode) %>%
    mutate(value = TRUE) %>%
    tidyr::pivot_wider(
      names_from = phecode,
      values_from = value,
      values_fill = list(value = FALSE)
    )
  
  # Ensure all original patients are preserved
  full_phenotypes <- merged %>%
    select(eid) %>%
    distinct() %>%
    left_join(phenotypes, by = "eid")
  
  # Optional: diagnostics
  full_phenotypes$phenotype_count <- rowSums(full_phenotypes[,-1], na.rm = TRUE)
  cat("Patients with ≥1 phenotype assigned:", sum(full_phenotypes$phenotype_count > 0), "\n")
  cat("Total patients:", nrow(full_phenotypes), "\n")
  full_phenotypes$phenotype_count <- NULL
  
  return(full_phenotypes)
}

# Update phenotypes using the function
phenotypes_result <- update_phenotypes(merged, phecode_map_icd10)

# Save the filtered dataset to a CSV file
write.csv(phenotypes_result, "/Users/philipcroon/PycharmProjects/Yale/DL_ECG_PheWAS/data/phenotypes_result_ukb.csv", row.names = FALSE)

# Load the filtered dataset from the CSV file only when running from here
phenotypes_result <- read.csv("/Users/philipcroon/PycharmProjects/Yale/DL_ECG_PheWAS/data/phenotypes_result_ukb.csv",
                              check.names = FALSE)
cat("Number of patients", nrow(phenotypes_result), "\n")
class(phenotypes_result)

min_cases <- 20  # Set this value once at the beginning

# Drop columns where all values are NA
drop_cols <- apply(phenotypes_result[, -1], 2, function(col) all(is.na(col)))
phenotypes_result <- phenotypes_result[, c(TRUE, !drop_cols)]

# Calculate counts of TRUE values for each phenotype
phenotype_counts <- colSums(phenotypes_result[, -1], na.rm = TRUE)

# Diagnostic: see the range of phenotype case counts
cat("Phenotype case count summary:\n")
print(summary(phenotype_counts))

# Filter to phenotypes with > min_cases TRUE values
keep_cols <- phenotype_counts > min_cases

# Diagnostic: how many are being kept
cat("Number of phenotypes kept:", sum(keep_cols), "\n")

# Subset the data
phenotypes_result_sel <- phenotypes_result[, c(TRUE, keep_cols)]

# Prepare additional dataframes for merging
id.sex <- data.frame(id = merged$eid, sex = merged$sex_f31_0_0)
id.age <- data.frame(id = merged$eid, age = merged$age_when_attended_assessment_centre_f21003_2_0)
phenotypes_result_sel <- phenotypes_result_sel %>% rename(id = eid)

# Initialize a list to store results for each model
all_results <- list()

# Loop over each AI model
for (model in models) {
  
  # Define genotypes for the current model
  genotypes <- data.frame(id = merged$eid, dassi = merged[[model]])
  
  data <- phenotypes_result_sel %>%
    left_join(genotypes, by = "id") %>%
    left_join(id.sex, by = "id") %>%
    left_join(id.age, by = "id")
  
  pheno_cols <- names(phenotypes_result_sel)[-1]
  
  # 1) How many patients have _no_ phecode data at all?
  missing_all <- rowSums(!is.na(data[pheno_cols])) == 0
  cat("Patients with zero phecodes (all NA):", sum(missing_all), "\n")
  if (sum(missing_all) > 0) {
    cat("Example IDs with no phecodes:", 
        paste(head(data$id[missing_all], 10), collapse = ", "), "\n")
  }
  
  # 2) Missingness _per_ phenotype
  missing_per_pheno <- sapply(pheno_cols, function(ph) sum(is.na(data[[ph]])))
  print(head(missing_per_pheno, 10))   # first 10 phenotypes
  cat("Range of NA counts per phecode:", range(missing_per_pheno), "\n")
  
  cat("Number of patients", nrow(data), "\n")
  
  # Define the covariates based on the model
  if (model == "preds_male_sex_model_unfrozen_ep5" || model == "sex_f31_0_0") {
    covariates <- c("age")  # Only age for these specific models
  } else {
    covariates <- c("sex", "age")  # Sex and age for other models
  }
  
  # Quick check: how many patients have complete data for ALL covariates + genotype?
  complete_cov <- data %>%
    filter(!is.na(dassi), !is.na(sex), !is.na(age))
  
  cat("Patients with complete genotype+covariates:", nrow(complete_cov), "\n")
  
  # Now for each phenotype about to be tested, see how many non-missing
  pheno_cols <- names(phenotypes_result_sel)[-1]
  missing_by_pheno <- sapply(pheno_cols, function(ph) {
    sum(!is.na(data[[ph]]))
  })
  summary(missing_by_pheno)
  cat("Range of non-missing phenotype counts:\n")
  print(range(missing_by_pheno))
  
  # Now run PheWAS
  results <- PheWAS::phewas(
    data,
    phenotypes  = pheno_cols,
    genotypes   = "dassi",
    covariates  = covariates,
    significance.threshold = "bonferroni",
    cores       = 1
  )
  
  # Compare to results$n_total
  cat("phewas() returned n_total range:",
      min(results$n_total), "to", max(results$n_total), "\n")
  
  # Generate the Manhattan plot with the model in the title
  phewas_plot <- PheWAS::phewasManhattan(results, 
                                         OR.direction = TRUE,
                                         y.axis.interval = 20,
                                         max.y =50,
                                         size.x.labels = 10,
                                         size.y.labels = 10,
                                         annotate.size = 4,
                                         title = paste("PheWAS Manhattan Plot for Model:", model))
  
  # Explicitly print the plot
  print(phewas_plot)
  ggsave(filename = paste0("/Users/philipcroon/PycharmProjects/Yale/DL_ECG_PheWAS/data/phewas_plot_", model, "_min", min_cases, ".pdf"),
         plot = phewas_plot, width = 10, height = 6, dpi = 300)

  # Add Phecode information to the results
  results_d <- PheWAS::addPhecodeInfo(results)
  
  # Store results in the list using the model name as the key
  all_results[[model]] <- results_d
  
  # Write each result to a CSV file
  write.csv(results_d, file = paste0("/Users/philipcroon/PycharmProjects/Yale/DL_ECG_PheWAS/results_", model, "_min", min_cases, ".csv"))
  
  # Optional: Print the top 10 results for each model
  cat("Top results for model:", model, "\n")
   print(results_d[order(results_d$p)[1:10], "phenotype"])
}

# Display a message indicating completion
cat("Analysis complete for all models.\n")

