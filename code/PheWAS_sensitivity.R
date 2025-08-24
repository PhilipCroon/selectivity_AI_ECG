rm(list = ls())
# Load necessary libraries
library(dplyr)
library(PheWAS)

# institutions <- c("YNHH", "Community", "NEMG")
institutions <- c("YNHH")

# Define a list of AI models you want to test
models <- c(
  "preds_image_ModerateOrSevereAS",
  "preds_image_HCM_LVDD_IVSd15_IntermediateAsFalse",
  "preds_image_ModerateOrSevereMR",
  "preds_image_Under40",
  "preds_image_ModerateOrSevereValveDisease",
  'preds_image_cmp_HCM_LVDD_ModtoSevVD_EF40',
  'preds_image_MaleSex'
)

# models <- c(
#   'preds_Dermatophytosis', 
#   'preds_image_EvenMonth', 
#   'preds_Bitten_by_dog', 
#   'preds_Fractures', 
#   'preds_Headache', 
#   'preds_Transport_accident', 
#   'preds_Viral_respiratory_infection'
#   )


# Load the yale data
yale <- read.delim("/Users/philipcroon/PycharmProjects/Yale/DL_ECG_PheWAS/data/Yale_ICD_long_format_random_all_hospitals.tsv", 
                   header = TRUE, sep = "\t", stringsAsFactors = FALSE)

# Add exp outcomes
exp <- read.delim("/Users/philipcroon/PycharmProjects/Yale/DL_ECG_PheWAS/data/exp_preds_ynhhs.csv",
                  header = TRUE, sep = ",", stringsAsFactors = FALSE)

# Merge on PAT_MRN_ID (inner join by default)
yale <- merge(yale, exp, by = "PAT_MRN_ID")

# # Create a new column 'Female' in the yale data frame
yale$Female <- ifelse(yale$SEX == "Female", 1, 0)


# First set of columns: eid and ICD-10 diagnosis codes
desired_columns_1 <- c("PAT_MRN_ID", "institution", grep("^ICD_code_", names(yale), value = TRUE))
# Second column to add: age information
desired_columns_2 <- "Age_at_ECG"

# Third set of columns: all possible AI model columns and sex information
desired_columns_3 <- c(
  "preds_image_HCM_LVDD_IVSd15_IntermediateAsFalse", 
  "preds_image_ModerateOrSevereAS", 
  "preds_image_ModerateOrSevereMR", 
  "preds_image_Under40", 
  "preds_image_ModerateOrSevereValveDisease", 
  'preds_image_cmp_HCM_LVDD_ModtoSevVD_EF40',
  'preds_image_HCM_LVDD_IVSd15_IntermediateAsFalse',
  'preds_image_MaleSex', 
  'preds_image_EvenMonth', 
  'preds_Dermatophytosis', 
  'preds_Bitten_by_dog', 
  'preds_Fractures', 
  'preds_Headache', 
  'preds_Transport_accident', 
  'preds_Viral_respiratory_infection',
  # 'preds_image_HCM_NatCVR',
  'Female')

# Combine all the columns into a single vector and remove duplicates
desired_columns <- unique(c(desired_columns_1, desired_columns_2, desired_columns_3))

missing_columns <- setdiff(desired_columns, colnames(yale))

# Print missing columns
if (length(missing_columns) > 0) {
  print("Missing columns:")
  print(missing_columns)
} else {
  print("All columns are present.")
}

merged <- yale[, desired_columns]

cat("Number of patients", nrow(merged), "\n")
# Remove rows with NaN or NA in any of the model columns
merged <- merged[!apply(merged[models], 1, function(row) any(is.na(row))), ]
cat("Number of patients", nrow(merged), "\n")

for (institution_name in institutions) {
  
  # Filter based on institution
  if (institution_name == "Community") {
    inst_df <- subset(merged, merged$institution %in% c('Westerly', 'L_and_M', 'Bridgeport', 'Greenwich'))
  } else {
    inst_df <- subset(merged, merged$institution == institution_name)
  }
  
  cat("Number of patients in", institution_name, "cohort:", nrow(inst_df), "\n")
  
  # Print missing columns
  if (length(missing_columns) > 0) {
    print("Missing columns:")
    print(missing_columns)
  } else {
    print("All columns are present.")
  }
  
  # -- Optional: Logit and standardize model predictions here if you want --
  for (model in models) {
    merged[[model]] <- log(merged[[model]] / (1 - merged[[model]]))
    merged[[model]] <- scale(merged[[model]], center = TRUE, scale = TRUE)
  }
  
  # Function to update phenotypes
  update_phenotypes <- function(inst_df, phecode_map_icd10) {
    # Remove dots from ICD codes
    phecode_map_icd10$code <- gsub("\\.", "", phecode_map_icd10$code)
    diagnosis_cols <- grep("^ICD_code", names(inst_df), value = TRUE)
    for (col in diagnosis_cols) {
      inst_df[[col]] <- gsub("\\.", "", inst_df[[col]])
    }
    
    # Convert to long format
    icd_long <- inst_df %>%
      select(PAT_MRN_ID, all_of(diagnosis_cols)) %>%
      tidyr::pivot_longer(
        cols = -PAT_MRN_ID,
        names_to = "icd_col",
        values_to = "icd_code"
      ) %>%
      filter(!is.na(icd_code), icd_code != "")
    
    # Map to phecodes
    icd_phecode <- icd_long %>%
      left_join(phecode_map_icd10, by = c("icd_code" = "code"), relationship = "many-to-many") %>%
      filter(!is.na(phecode), phecode != "")
    
    # Create wide phenotype matrix
    phenotypes <- icd_phecode %>%
      distinct(PAT_MRN_ID, phecode) %>%
      mutate(value = TRUE) %>%
      tidyr::pivot_wider(
        names_from = phecode,
        values_from = value,
        values_fill = list(value = FALSE)
      )
    
    # Merge with all patients
    full_phenotypes <- inst_df %>%
      select(PAT_MRN_ID) %>%
      distinct() %>%
      left_join(phenotypes, by = "PAT_MRN_ID")
    
    # Diagnostics
    full_phenotypes$phenotype_count <- rowSums(full_phenotypes[ , -1], na.rm = TRUE)
    cat("Patients with ≥1 phenotype assigned:", sum(full_phenotypes$phenotype_count > 0), "\n")
    cat("Total patients:", nrow(full_phenotypes), "\n")
    
    # Drop helper
    full_phenotypes$phenotype_count <- NULL
    
    return(full_phenotypes)
  }
  
  cat("Number of patients 1", nrow(merged), "\n")
  
  # Update phenotypes
  phenotypes_result <- update_phenotypes(inst_df, phecode_map_icd10)
  
  cat("Number of patients 2", nrow(phenotypes_result), "\n")
  
  # Save results
  write.csv(
    phenotypes_result, 
    file = paste0("/Users/philipcroon/PycharmProjects/Yale/DL_ECG_PheWAS/data/phenotypes_result_sel_", institution_name, ".csv"), 
    row.names = FALSE
  )
}

cat("so far so good")
for (institution_name in institutions) {
  
  # Load the filtered dataset from the CSV file only when running from here
  phenotypes_result <- read.csv(paste0("/Users/philipcroon/PycharmProjects/Yale/DL_ECG_PheWAS/data/phenotypes_result_sel_", 
                                       institution_name, ".csv"),
                                check.names = FALSE)
  cat("Number of patients", nrow(phenotypes_result), "\n")
  
  # Set your threshold
  min_cases <- 20
  
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
  
  id.sex <- data.frame(PAT_MRN_ID = merged$PAT_MRN_ID, Female = merged$Female)
  id.age <- data.frame(PAT_MRN_ID = merged$PAT_MRN_ID, age = merged$Age_at_ECG)
  
  phenotypes_result_sel <- phenotypes_result_sel %>%
    inner_join(id.sex, by = "PAT_MRN_ID") %>%
    inner_join(id.age, by = "PAT_MRN_ID")
  
  phenotype_names <- names(phenotypes_result_sel)[2:(ncol(phenotypes_result_sel) - 4)]
  
  # —————— START REPLICATION FILTER ——————
# after your two inner_join()s, before filtering
cat("👉 Rows after joins, before filter: ", nrow(phenotypes_result_sel), "\n")

# now do the “no-op” filter
selected_phecodes <- c("428.2", "394.3", "394.2", "416")
phenotypes_result_sel <- phenotypes_result_sel %>%
  filter(
    # coerce logical→integer and ignore NAs just in case
    rowSums(across(all_of(selected_phecodes), ~ as.integer(.x)), na.rm=TRUE)
    <= 1
  )

cat("👉 Rows after no-op filter (≤4):       ", nrow(phenotypes_result_sel), "\n")
  
  # —————— END REPLICATION FILTER ——————
  
  # Initialize a list to store results for each model
  all_results <- list()
  
  # # Loop over each AI model
  for (model in models) {
    
    data <- phenotypes_result_sel %>%
      inner_join(merged %>% select(PAT_MRN_ID, all_of(model)) %>% rename(dassi = all_of(model)),
                 by = "PAT_MRN_ID")
    
    # Define the covariates based on the model
    if (model == 'preds_image_MaleSex') {
      covariates <- c("age")  # Only age for this specific model
      
      # Drop multiple columns
      
    } else {
      covariates <- c("Female", "age")  # Sex and age for other models
    }
    
    results <- PheWAS::phewas(
      data,
      phenotypes = phenotype_names,
      genotypes = c("dassi"),
      significance.threshold = "bonferroni",
      covariates = covariates,
      additive.genotypes = FALSE,  # Treat genotype as categorical
      cores = 1
    )
    
    # Generate the Manhattan plot with the model in the title
    phewas_plot <- PheWAS::phewasManhattan(results,
                                           OR.direction = TRUE,
                                           y.axis.interval = 20,
                                           max.y =100,
                                           size.x.labels = 10,
                                           size.y.labels = 10,
                                           annotate.size = 4,
                                           title = paste("PheWAS Manhattan Plot for Model:", model))
    
    # Explicitly print the plot
    print(phewas_plot)
    ggsave(filename = paste0("/Users/philipcroon/PycharmProjects/Yale/DL_ECG_PheWAS/data/SENS1yale_phewas_plot_",
                             model, "_", institution_name, "_min", min_cases, ".pdf"),
           plot = phewas_plot, width = 10, height = 6, dpi = 300)
    
    # Add Phecode information to the results
    results_d <- PheWAS::addPhecodeInfo(results)
    
    # Store results in the list using the model name as the key
    all_results[[model]] <- results_d
    # Write each result to a CSV file
    write.csv(results_d,
              file = paste0("/Users/philipcroon/PycharmProjects/Yale/DL_ECG_PheWAS/SENS1yale_results_",
                            model, "_", institution_name, "_min", min_cases, ".csv"))
    
    #
    #   # Optional: Print the top 10 results for each model
    #   cat("Top results for model:", model, "\n")
    #   print(results_d[order(results_d$p)[1:10], "phenotype"])
  }
}

# Display a message indicating completion
cat("Analysis complete for all models.\n")

