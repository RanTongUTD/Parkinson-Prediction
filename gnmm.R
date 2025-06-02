# --- Load necessary libraries ---
# Ensure lme4 package is installed and loaded if Network_Functions.R requires it
# install.packages("lme4") # Run once if not installed
library(lme4) # Load it for the session

# --- Define URL and Column Names ---
data_url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data"
# Define column names based on UCI description
col_names <- c("subject#", "age", "sex", "test_time", "motor_UPDRS", "total_UPDRS",
               "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
               "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11",
               "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE")

# --- Load data from URL ---
print(paste("Loading data from:", data_url))
full_data <- tryCatch({
  # Ensure check.names = FALSE is included to preserve original names
  read.csv(data_url, header = FALSE, col.names = col_names, stringsAsFactors = FALSE, check.names = FALSE)
}, error = function(e) {
  stop("Failed to load data from URL. Error: ", e$message)
})
print("Data loaded successfully.")

# --- Check if columns loaded correctly AFTER loading data ---
if (!"subject#" %in% names(full_data)) stop("Subject column 'subject#' not found after loading. Check check.names=FALSE.")
if (!"total_UPDRS" %in% names(full_data)) stop("Outcome column 'total_UPDRS' not found after loading.")

# --- Define outcome, subject, and explicitly select ONLY numeric voice measure predictors ---
outcome_col <- 'total_UPDRS'
subject_col <- 'subject#'

# Explicitly define predictors as the 16 voice measures
predictor_cols <- c(
  "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
  "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11",
  "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE"
)

# --- Verify predictor columns exist ---
missing_predictors <- setdiff(predictor_cols, names(full_data))
if (length(missing_predictors) > 0) stop("Defined predictor columns not found: ", paste(missing_predictors, collapse=", "))
print(paste("Using", length(predictor_cols), "explicitly selected predictor columns."))


# --- Create outcome vector (Y) and predictor matrix (X) ---
Y_full <- full_data[[outcome_col]]
X_full <- full_data[, predictor_cols, drop = FALSE] # Select only the 16 predictors
lab_full <- full_data[[subject_col]]

# --- Ensure data types are numeric ---
# ***** CRITICAL FIX: Reinstate loop to convert selected X columns to numeric *****
print("Converting selected predictor columns to numeric...")
for (col in predictor_cols) {
  # Check if column exists before trying to convert (should always exist based on check above)
  if (col %in% names(X_full)) {
    # Important: Use as.numeric(as.character(...)) for robustness if data was read as factor/character
    X_full[[col]] <- suppressWarnings(as.numeric(as.character(X_full[[col]])))
  }
}
print("Predictor columns conversion attempted.")

# Check/convert outcome Y
print("Checking and converting outcome type to numeric...")
if (!is.numeric(Y_full)) {
  warning("Converting non-numeric outcome column.")
  Y_full <- suppressWarnings(as.numeric(as.character(Y_full)))
} else {
  print("Outcome column is numeric.")
}

# --- Optional: Check types again after conversion ---
print("Checking predictor types AFTER conversion...")
non_numeric_after_conv <- predictor_cols[!sapply(X_full, is.numeric)]
if (length(non_numeric_after_conv) > 0) {
  warning("ISSUE persists: Predictors still not numeric after conversion: ", paste(non_numeric_after_conv, collapse=", "))
} else {
  print("Selected predictor columns are now numeric.")
}


# --- Remove observations with NA's (in Y or X, potentially introduced by conversion) ---
print("Checking for NA values after conversions...")
rmX <- which(apply(X_full, 1, function(row) any(is.na(row))))
rmY <- which(is.na(Y_full))
rmXY <- unique(c(rmX, rmY))

if (length(rmXY) == nrow(X_full)) {
  stop("All rows seem to contain NA values after conversions.")
}

if (length(rmXY) > 0) {
  print(paste("Removing", length(rmXY), "rows containing NA values (original or from conversion)."))
  lab <- lab_full[-rmXY]
  Y <- Y_full[-rmXY]
  X_unscaled <- X_full[-rmXY, , drop = FALSE]
} else {
  print("No rows with NA values found after conversions.")
  lab <- lab_full
  Y <- Y_full
  X_unscaled <- X_full
}

# --- Check if data remains ---
if (nrow(X_unscaled) == 0 || length(Y) == 0 || length(lab) == 0) stop("No data remaining after NA removal.")
if (length(unique(lab)) < 2) stop("Fewer than 2 subjects remaining.")

# --- Scale predictors ---
# This should now work because X_unscaled had non-numeric columns converted
print("Scaling predictors...")
X <- scale(as.matrix(X_unscaled)) # Input to scale() MUST be numeric
print("Predictors scaled.")

# --- Split data into training and testing ---
print("Splitting data into training and testing sets...")
unique_subjects <- unique(lab)
m <- length(unique_subjects)

if (m == 0) stop("No subjects found after processing.")

n_obs_per_subject <- sapply(unique_subjects, function(subj_id) sum(lab == subj_id))
if(any(n_obs_per_subject <= 0)) stop("Error calculating observations per subject.")

cumulative_n <- cumsum(n_obs_per_subject)
last.obs.indices <- cumulative_n

if (max(last.obs.indices) > length(Y)) stop("Calculated indices exceed total observations.")
if (length(last.obs.indices) != m) stop("Mismatch between subjects and last observation indices.")

Y.train <- Y[-last.obs.indices]
X.train <- X[-last.obs.indices, , drop = FALSE]
lab.train <- lab[-last.obs.indices]

Y.test <- Y[last.obs.indices]
X.test <- X[last.obs.indices, , drop = FALSE]
lab.test <- lab[last.obs.indices]

if (length(Y.train) == 0 || nrow(X.train) == 0) stop("Training set is empty after splitting.")
if (length(Y.test) == 0 || nrow(X.test) == 0) stop("Test set is empty after splitting.")
if (length(Y.test) != m) warning(paste("Num test samples (", length(Y.test), ") != num subjects (", m, ")."))

print(paste("Data split:", length(Y.train), "training points,", length(Y.test), "testing points."))


# --- Run 1-layer GNMM Model ---
# IMPORTANT: Set working directory correctly or use full path for source()
# setwd("/Users/rantong/Downloads/biom13615-sup-0001-suppmat/SuppMat") # Your path
print("Sourcing Network_Functions.R...")
source('Network_Functions.R') # Assumes file is in the working directory set above
print("Network_Functions.R sourced.")

# --- Network settings ---
nepochs <- 2 # Still might be slow; reduce to e.g., 5 for quick testing
hidnodes1 <- 3
num_runs <- 5
set.seed(12044)
mspe.gnmm1 <- rep(NA, num_runs)

# Use the prepared training data directly
Y.train_run <- Y.train
X.train_run <- X.train
lab.train_run <- lab.train

print(paste("Starting 1-layer GNMM training for", num_runs, "runs with", nepochs, "epochs..."))

for (i in 1:num_runs) {
  print(paste("Running GNMM iteration:", i))
  start_time <- Sys.time()
  
  m1 <- tryCatch({
    gnmm.sgd(formula = Y.train_run ~ X.train_run + (1|lab.train_run),
             family = 'gaussian',
             penalization = 0.001,
             nodes1 = hidnodes1,
             nodes2 = NULL,
             step_size = 0.005,
             act_fun = 'relu',
             nepochs = nepochs,
             incl_ranef = TRUE)
  }, error = function(e) {
    warning(paste("Error during gnmm.sgd run", i, ":", conditionMessage(e)))
    NULL
  })
  
  end_time <- Sys.time()
  duration <- end_time - start_time
  print(paste("gnmm.sgd call finished for iteration", i, ". Duration:", round(duration, 2), units(duration)))
  
  if (!is.null(m1)) {
    pred1 <- tryCatch({
      gnmm.predict(new_data = X.test, id = lab.test, gnmm.fit = m1)
    }, error = function(e) {
      warning(paste("Error during gnmm.predict run", i, ":", conditionMessage(e)))
      NULL
    })
    
    if (!is.null(pred1)) {
      if (length(pred1) == length(Y.test)) {
        if(any(is.nan(pred1)) || any(is.infinite(pred1))) {
          warning(paste("NaN or Inf values found in predictions for run", i, ". MSPE will be NA."))
          mspe.gnmm1[i] <- NA
        } else {
          mspe.gnmm1[i] <- mean((Y.test - pred1)^2, na.rm = TRUE)
        }
      } else {
        warning(paste("Prediction length mismatch run", i, "."))
      }
    } else { warning(paste("Skipping MSPE calc run", i, ".")) }
  } else { warning(paste("Skipping prediction & MSPE calc run", i, ".")) }
  
  print(paste("Finished iteration:", i, "- Current MSPE:", round(mspe.gnmm1[i], 4)))
}

# --- Display Results ---
print("--------------------")
print("Final MSPE results from the runs:")
print(round(mspe.gnmm1, 4))

average_mspe <- mean(mspe.gnmm1, na.rm = TRUE)
if (is.nan(average_mspe)) {
  print("Average MSPE could not be calculated.")
} else {
  print(paste("Average MSPE for 1-layer GNMM:", round(average_mspe, 4)))
}
print("--------------------")











#############run ann for comparison
# --- Run ANN Model ---
# This uses the same gnmm.sgd function but disables the random effects part.

print("Starting ANN training...")

# Network settings (assuming nepochs and hidnodes1 are already defined)
# nepochs <- 50 # Or use the value already set
# hidnodes1 <- 3 # Or use the value already set

# Use a different seed than GNMM
set.seed(12045)
num_runs <- 3
mspe.ann <- rep(NA, num_runs)
mae.ann <- rep(NA, num_runs) # <<< Also add MAE calculation here if desired

# Use the same prepared training data
Y.train_run <- Y.train
X.train_run <- X.train
lab.train_run <- lab.train

print(paste("Starting ANN training for", num_runs, "runs with", nepochs, "epochs..."))

for (i in 1:num_runs) {
  print(paste("Running ANN iteration:", i))
  start_time <- Sys.time()
  
  m3 <- tryCatch({
    gnmm.sgd(formula = Y.train_run ~ X.train_run + (1|lab.train_run),
             family = 'gaussian',
             penalization = 0.001,
             nodes1 = hidnodes1,
             nodes2 = NULL,
             # ***** TRY REDUCING step_size *****
             step_size = 0.001, # Original was 0.005. Try 0.001 or 0.0005
             # **********************************
             act_fun = 'relu',
             nepochs = nepochs,
             incl_ranef = FALSE) # <<< Key difference for ANN
  }, error = function(e) {
    warning(paste("Error during ANN (gnmm.sgd) run", i, ":", conditionMessage(e)))
    NULL
  })
  
  end_time <- Sys.time()
  duration <- end_time - start_time
  print(paste("ANN (gnmm.sgd) call finished for iteration", i, ". Duration:", round(duration, 2), units(duration)))
  
  # Predict on the test data if model fitting succeeded
  if (!is.null(m3)) {
    pred3 <- tryCatch({
      gnmm.predict(new_data = X.test, id = lab.test, gnmm.fit = m3)
    }, error = function(e) {
      warning(paste("Error during ANN prediction run", i, ":", conditionMessage(e)))
      NULL
    })
    
    # Calculate Metrics if prediction succeeded
    if (!is.null(pred3)) {
      if (length(pred3) == length(Y.test)) {
        if(any(is.nan(pred3)) || any(is.infinite(pred3))) {
          warning(paste("NaN or Inf values found in ANN predictions for run", i, ". Metrics will be NA."))
          mspe.ann[i] <- NA
          mae.ann[i] <- NA # <<< Also set MAE to NA
        } else {
          mspe.ann[i] <- mean((Y.test - pred3)^2, na.rm = TRUE)
          mae.ann[i] <- mean(abs(Y.test - pred3), na.rm = TRUE) # <<< Calculate MAE
        }
      } else {
        warning(paste("ANN Prediction length mismatch run", i, "."))
      }
    } else { warning(paste("Skipping ANN Metrics calc run", i, ".")) }
  } else { warning(paste("Skipping ANN prediction & Metrics calc run", i, ".")) }
  
  # Print both metrics
  print(paste("Finished ANN iteration:", i, "- MSPE:", round(mspe.ann[i], 4), "- MAE:", round(mae.ann[i], 4)))
}

# --- Display ANN Results ---
print("--------------------")
print("Final ANN MSPE (Test MSE) results:")
print(round(mspe.ann, 4))
print("Final ANN Test MAE results:")
print(round(mae.ann, 4)) # <<< Print MAE results

average_mspe_ann <- mean(mspe.ann, na.rm = TRUE)
average_mae_ann <- mean(mae.ann, na.rm = TRUE) # <<< Calculate average MAE

if (is.nan(average_mspe_ann)) {
  print("Average ANN MSPE could not be calculated.")
} else {
  print(paste("Average MSPE for ANN:", round(average_mspe_ann, 4)))
}
if (is.nan(average_mae_ann)) { # <<< Report average MAE
  print("Average ANN MAE could not be calculated.")
} else {
  print(paste("Average MAE for ANN:", round(average_mae_ann, 4)))
}









########we need the 1 layer gnmm to show mae as well:
# --- Run 1-layer GNMM Model & Calculate BOTH MSE and MAE ---
# IMPORTANT: Set working directory correctly or use full path for source()
# setwd("/path/to/your/folder") # Example
print("Sourcing Network_Functions.R...")
source('Network_Functions.R') # Assumes file is in the working directory set above
print("Network_Functions.R sourced.")

# --- Network settings ---
nepochs <- 2 # Set desired epochs (e.g., 50 for full run, or less for test)
hidnodes1 <- 3
num_runs <- 5
set.seed(12044)

# --- Create vectors to store results ---
mspe.gnmm1 <- rep(NA, num_runs)
mae.gnmm1 <- rep(NA, num_runs) # <<< Added vector for MAE

# Use the prepared training data directly
# Assumes Y.train, X.train, lab.train, Y.test, X.test, lab.test exist from previous steps
Y.train_run <- Y.train
X.train_run <- X.train
lab.train_run <- lab.train

print(paste("Starting 1-layer GNMM training for", num_runs, "runs with", nepochs, "epochs..."))

for (i in 1:num_runs) {
  print(paste("Running GNMM iteration:", i))
  start_time <- Sys.time()
  
  m1 <- tryCatch({
    gnmm.sgd(formula = Y.train_run ~ X.train_run + (1|lab.train_run),
             family = 'gaussian',
             penalization = 0.001,
             nodes1 = hidnodes1,
             nodes2 = NULL,
             step_size = 0.005,
             act_fun = 'relu',
             nepochs = nepochs,
             incl_ranef = TRUE)
  }, error = function(e) {
    warning(paste("Error during gnmm.sgd run", i, ":", conditionMessage(e)))
    NULL
  })
  
  end_time <- Sys.time()
  duration <- end_time - start_time
  print(paste("gnmm.sgd call finished for iteration", i, ". Duration:", round(duration, 2), units(duration)))
  
  # --- Calculate Metrics if model ran successfully ---
  if (!is.null(m1)) {
    pred1 <- tryCatch({
      gnmm.predict(new_data = X.test, id = lab.test, gnmm.fit = m1)
    }, error = function(e) {
      warning(paste("Error during gnmm.predict run", i, ":", conditionMessage(e)))
      NULL
    })
    
    if (!is.null(pred1)) {
      if (length(pred1) == length(Y.test)) {
        # Check for NaN/Inf in predictions before calculating metrics
        if(any(is.nan(pred1)) || any(is.infinite(pred1))) {
          warning(paste("NaN or Inf values found in predictions for run", i, ". Metrics will be NA."))
          mspe.gnmm1[i] <- NA
          mae.gnmm1[i] <- NA # <<< Set MAE to NA too
        } else {
          # Calculate MSPE (MSE on Test)
          mspe.gnmm1[i] <- mean((Y.test - pred1)^2, na.rm = TRUE)
          # Calculate MAE (Mean Absolute Error on Test) # <<< Added MAE calculation
          mae.gnmm1[i] <- mean(abs(Y.test - pred1), na.rm = TRUE)
        }
      } else {
        warning(paste("Prediction length mismatch run", i, "."))
      }
    } else { warning(paste("Skipping Metrics calc run", i, ".")) }
  } else { warning(paste("Skipping prediction & Metrics calc run", i, ".")) }
  
  # Print both metrics for the current iteration
  print(paste("Finished iteration:", i, "- MSPE:", round(mspe.gnmm1[i], 4), "- MAE:", round(mae.gnmm1[i], 4)))
}

# --- Display Results (including MAE) ---
print("--------------------")
print("Final MSPE (Test MSE) results for 1-layer GNMM:")
print(round(mspe.gnmm1, 4))
print("Final Test MAE results for 1-layer GNMM:")
print(round(mae.gnmm1, 4)) # <<< Print MAE results

average_mspe <- mean(mspe.gnmm1, na.rm = TRUE)
average_mae <- mean(mae.gnmm1, na.rm = TRUE) # <<< Calculate average MAE

if (is.nan(average_mspe)) {
  print("Average MSPE could not be calculated.")
} else {
  print(paste("Average MSPE for 1-layer GNMM:", round(average_mspe, 4)))
}
if (is.nan(average_mae)) { # <<< Report average MAE
  print("Average MAE could not be calculated.")
} else {
  print(paste("Average MAE for 1-layer GNMM:", round(average_mae, 4)))
}
print("--------------------")
print("--------------------")







#################run the 2 layer example

# --- Run 2-layer GNMM Model & Calculate BOTH MSE and MAE ---

print("Starting 2-layer GNMM training...")

# --- Network settings (assuming hidnodes1 is already defined) ---
# hidnodes1 <- 3 # Should exist from previous model runs
hidnodes2 <- 2 # Define nodes for the second hidden layer based on original script
nepochs <- 5 # Set desired epochs (e.g., 50 for full run, or less for test)
num_runs <- 3
set.seed(12045) # Seed used in the original script for 2-layer GNMM

# --- Create vectors to store results ---
mspe.gnmm2 <- rep(NA, num_runs)
mae.gnmm2 <- rep(NA, num_runs) # Added vector for MAE

# Use the prepared training data directly
# Assumes Y.train, X.train, lab.train, Y.test, X.test, lab.test exist
Y.train_run <- Y.train
X.train_run <- X.train
lab.train_run <- lab.train

print(paste("Starting 2-layer GNMM training for", num_runs, "runs with", nepochs, "epochs..."))

for (i in 1:num_runs) {
  print(paste("Running 2-layer GNMM iteration:", i))
  start_time <- Sys.time()
  
  m2 <- tryCatch({
    gnmm.sgd(formula = Y.train_run ~ X.train_run + (1|lab.train_run),
             family = 'gaussian',
             penalization = 0.002, # Penalization used in original 2-layer model
             nodes1 = hidnodes1,
             nodes2 = hidnodes2,   # <<< Use second layer nodes
             step_size = 0.005,   # Step size was same as 1-layer in original
             act_fun = 'relu',
             nepochs = nepochs,
             incl_ranef = TRUE)   # Include random effects
  }, error = function(e) {
    warning(paste("Error during 2-layer gnmm.sgd run", i, ":", conditionMessage(e)))
    NULL
  })
  
  end_time <- Sys.time()
  duration <- end_time - start_time
  print(paste("2-layer gnmm.sgd call finished for iteration", i, ". Duration:", round(duration, 2), units(duration)))
  
  # --- Calculate Metrics if model ran successfully ---
  if (!is.null(m2)) {
    pred2 <- tryCatch({
      gnmm.predict(new_data = X.test, id = lab.test, gnmm.fit = m2)
    }, error = function(e) {
      warning(paste("Error during 2-layer gnmm.predict run", i, ":", conditionMessage(e)))
      NULL
    })
    
    if (!is.null(pred2)) {
      if (length(pred2) == length(Y.test)) {
        # Check for NaN/Inf in predictions before calculating metrics
        if(any(is.nan(pred2)) || any(is.infinite(pred2))) {
          warning(paste("NaN or Inf values found in 2-layer predictions for run", i, ". Metrics will be NA."))
          mspe.gnmm2[i] <- NA
          mae.gnmm2[i] <- NA # Set MAE to NA too
        } else {
          # Calculate MSPE (MSE on Test)
          mspe.gnmm2[i] <- mean((Y.test - pred2)^2, na.rm = TRUE)
          # Calculate MAE (Mean Absolute Error on Test) # Added MAE calculation
          mae.gnmm2[i] <- mean(abs(Y.test - pred2), na.rm = TRUE)
        }
      } else {
        warning(paste("2-layer Prediction length mismatch run", i, "."))
      }
    } else { warning(paste("Skipping 2-layer Metrics calc run", i, ".")) }
  } else { warning(paste("Skipping 2-layer prediction & Metrics calc run", i, ".")) }
  
  # Print both metrics for the current iteration
  print(paste("Finished 2-layer iteration:", i, "- MSPE:", round(mspe.gnmm2[i], 4), "- MAE:", round(mae.gnmm2[i], 4)))
}

# --- Display Results (including MAE) ---
print("--------------------")
print("Final MSPE (Test MSE) results for 2-layer GNMM:")
print(round(mspe.gnmm2, 4))
print("Final Test MAE results for 2-layer GNMM:")
print(round(mae.gnmm2, 4)) # Print MAE results

average_mspe_2layer <- mean(mspe.gnmm2, na.rm = TRUE)
average_mae_2layer <- mean(mae.gnmm2, na.rm = TRUE) # Calculate average MAE

if (is.nan(average_mspe_2layer)) {
  print("Average 2-layer MSPE could not be calculated.")
} else {
  print(paste("Average MSPE for 2-layer GNMM:", round(average_mspe_2layer, 4)))
}
if (is.nan(average_mae_2layer)) { # Report average MAE
  print("Average 2-layer MAE could not be calculated.")
} else {
  print(paste("Average MAE for 2-layer GNMM:", round(average_mae_2layer, 4)))
}
print("--------------------")