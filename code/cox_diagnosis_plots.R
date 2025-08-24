rm(list = ls())

# install.packages(c("readr","stringr","dplyr","survival","rms","survminer","ggplot2", "tidyverse"))

# ─── Directories ─────────────────────────────────────────────────
in_dir    <- "/Users/philipcroon/Downloads"
out_plots <- "/Users/philipcroon/Downloads"
dir.create(out_plots, showWarnings = FALSE, recursive = TRUE)

# ─── Find CSVs ────────────────────────────────────────────────────
files <- list.files(in_dir, pattern="^cox_df_.*\\.csv$", full.names = TRUE)
if (length(files)==0) stop("❌ No files found in ", in_dir)

# ─── Cohort mapping & order ──────────────────────────────────────
cohort_map <- c(
  YNHH      = "Yale New Haven Hospital",
  UKB       = "UK Biobank",
  NEMG      = "Outpatient clinics",
  Community = "Community hospitals"
)
site_levels <- names(cohort_map)

# ─── Predictor mapping ───────────────────────────────────────────
preds_map_yale <- c(
  preds_image_HCM_LVDD_IVSd15_IntermediateAsFalse = "Left ventricular Hypertrophy",
  preds_image_ModerateOrSevereAS                  = "Moderate or Severe Aortic Stenosis",
  preds_image_ModerateOrSevereMR                  = "Moderate or Severe Mitral Regurgitation",
  preds_image_Under40                             = "Left Ventricular Systolic Dysfunction",
  preds_image_ModerateOrSevereValveDisease        = "Moderate or Severe Valvular Heart Disease",
  preds_image_cmp_HCM_LVDD_ModtoSevVD_EF40        = "Structural Heart Disease",
  preds_image_HCM_NatCVR                          = "Hypertrophic Cardiomyopathy",
  preds_image_MaleSex                             = "Male Sex"
)

# ─── Outcome mapping ─────────────────────────────────────────────
outcome_label_map <- c(
  "Heart failure NOS"   = "Heart failure",
  "Aortic valve disease"= "Aortic valve disease",
  "Mitral valve disease"= "Mitral valve disease",
  "Cardiomegaly"        = "Left ventricular hypertropy",
  "Composite"          = "Composite"
)

# ─── Time/event vars ─────────────────────────────────────────────
time_var  <- "survival_time"
event_var <- "event"

# ─── Build file info & enforce ordering ──────────────────────────
site_levels      <- names(cohort_map)
# And optionally reverse them if that was desired:
# site_levels     <- rev(site_levels)

file_info <- tibble(path = files) %>%
  mutate(
    bn          = basename(path),
    parts       = str_split(gsub("^cox_df_|\\.csv$", "", bn), "_"),
    description = map_chr(parts, ~ .x[1]),                  # raw outcome
    cohort      = map_chr(parts, ~ tail(.x,1)),             # code
    indep_var   = map_chr(parts, ~ paste(.x[2:(length(.x)-1)], collapse = "_"))
  ) %>%
  filter(
    cohort       %in% site_levels,
    indep_var    %in% names(preds_map_yale),
    description  %in% names(outcome_label_map)
  ) %>%
  mutate(
    cohort           = factor(cohort, levels = site_levels),
    cohort_name      = cohort_map[as.character(cohort)],
    pretty_var       = preds_map_yale[indep_var],
    pretty_outcome   = outcome_label_map[description]
  ) %>%
  # NOW arrange by cohort → model → outcome 
  arrange(cohort, indep_var, description)

# ─── Generate and collect plots ──────────────────────────────────
all_plots <- list()
for (i in seq_len(nrow(file_info))) {
  row <- file_info[i, ]
  df  <- read_csv(row$path, show_col_types = FALSE) %>% select(-starts_with("..."))
  var <- row$indep_var
  pv  <- row$pretty_var
  po  <- row$pretty_outcome
  site<- row$cohort_name
  
  adjustors <- setdiff(names(df), c(var, time_var, event_var))
  
  # 1) Linear CoxPH
  fit_lin <- coxph(
    as.formula(paste0("Surv(", time_var, ",", event_var, ") ~ ", var,
                      if(length(adjustors)) paste0(" + ", paste(adjustors, collapse="+")) else "")),
    data = df, ties = "breslow"
  )
  
  # 1) Get cox.zph object
  zph <- cox.zph(fit_lin, transform = "rank")
  
  # 2) Build a data.frame of time (in years) and residual for your predictor
  res_df <- data.frame(
    time =   zph$time,           # already in years
    resid =  zph$y[, var]        # pull the column by name
  ) %>%
    filter(time <= 5)            # keep only 0–5 years
  
  # 3) Plot
  p1 <- ggplot(res_df, aes(x = time, y = resid)) +
    geom_point(size = 0.6, alpha = 0.6) +
    geom_smooth(method = "loess", se = FALSE, color = "steelblue") +
    scale_x_continuous(breaks = 0:5, limits = c(0, 5)) +
    labs(
      title    = paste("Model:", pv),
      subtitle = paste("Outcome:", po, "\nCohort:", site),
      x        = "Years since baseline",
      y        = "Scaled Schoenfeld residual"
    ) +
    theme_minimal(base_size = 8)
  
  all_plots <- c(all_plots, list(p1))
  
  # 3) Spline CoxPH with custom knots
  knots   <- quantile(df[[var]], probs = c(0.15,0.5,0.85), na.rm = TRUE)
  dd      <- datadist(df); options(datadist="dd")
  fit_spl <- cph(
    as.formula(paste0("Surv(", time_var, ",", event_var, ") ~ rcs(", var,
                      ", c(", paste(round(knots,3), collapse=","), "))",
                      if(length(adjustors)) paste0(" + ", paste(adjustors, collapse="+")) else "")),
    data = df, x=TRUE, y=TRUE, surv=TRUE, ties="breslow"
  )
  
  grid_vals <- seq(min(df[[var]],na.rm=TRUE), max(df[[var]],na.rm=TRUE), length=100)
  pred_df   <- do.call(Predict, c(
    list(fit_spl),
    setNames(list(grid_vals), var),
    list(fun=exp, ref.zero=TRUE, conf.int=0.95)
  )) %>% as.data.frame()
  
  p2 <- ggplot(pred_df, aes_string(x = var, y = "yhat")) +
    geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2) +
    geom_line(size = 0.5) +
    geom_hline(yintercept = 1, linetype = "dashed") +
    scale_y_log10(breaks = c(0.5,1,2,4,8)) +
    labs(
      title    = paste("Model:", pv),
      subtitle = paste("Outcome:", po, "\nCohort:", site),
      x        = paste0(pv, " (z-score)"),
      y        = "Hazard ratio"
    ) +
    theme_minimal(base_size = 8)
  all_plots <- c(all_plots, list(p2))
}

# ─── Draw dynamic 2-column grid ───────────────────────────────────
n_models <- nrow(file_info)
# paginate: 6 plots per page (2 columns × 3 rows)
plots_per_page <- 6
total_plots    <- length(all_plots)
n_pages        <- ceiling(total_plots / plots_per_page)

pdf(
  file   = file.path(out_plots, "combined_by_site_mapped.pdf"),
  width  = 8,
  height = 3 * 3        # 3 rows × ~3" per row = 9"
)

for (page in seq_len(n_pages)) {
  start_idx <- (page - 1) * plots_per_page + 1
  end_idx   <- min(page * plots_per_page, total_plots)
  
  grid.arrange(
    grobs = all_plots[start_idx:end_idx],
    ncol  = 2
  )
  # after grid.arrange, pdf() automatically moves to next page
}

dev.off()

message("✅ Saved final grid with outcome mapping to combined_by_site_mapped.pdf")
