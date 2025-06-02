library(readr)
library(dplyr)
library(ggplot2)
library(lme4)      

url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data"
data <- read_csv(url)

colnames(data) <- c("subject", "age", "sex", "test_time", "motor_UPDRS", "total_UPDRS", 
                             "Jitter_percent", "Jitter_Abs", "Jitter_RAP", "Jitter_PPQ5", "Jitter_DDP",
                             "Shimmer", "Shimmer_dB", "Shimmer_APQ3", "Shimmer_APQ5", "Shimmer_APQ11", "Shimmer_DDA",
                             "NHR", "HNR", "RPDE", "DFA", "PPE")
cat("Dimension:", dim(data), "\n")
data %>% select(subject, age, sex, test_time, total_UPDRS) %>% head()

# =======================================================================================
# Exploratory Data Analysis
dim(data) # 5875*22
length(unique(data$subject)) # 42 patients
str(data)
head(data)
colSums(is.na(data))  # no miss value

summary(data)
table(data$subject)   
table(data$sex)   # 0: 4008  1: 1867

# ----------------------------------------
# Distribution of variables
# total_UPDRS
ggplot(data, aes(x = total_UPDRS)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
  ggtitle("Distribution of total_UPDRS")

# age
ggplot(data, aes(x = age)) +
  geom_histogram(binwidth = 1, fill = "lightgreen", color = "black") +
  ggtitle("Distribution of Age")

# ----------------------------------------
# Trend of UPDRS along time
ggplot(subset(data, subject == 1), aes(x = test_time, y = total_UPDRS)) +
  geom_line(color = "blue") +
  ggtitle("Subject 1: total_UPDRS over Time")

# all subjects
ggplot(data, aes(x = test_time, y = total_UPDRS, group = subject)) +
  geom_line(alpha = 0.3) +
  ggtitle("UPDRS Progression for All Subjects")

# age vs total_UPDRS
ggplot(data, aes(x = age, y = total_UPDRS)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", color = "red") +
  ggtitle("Age vs total_UPDRS")

# RPDE vs total_UPDRS
ggplot(data, aes(x = RPDE, y = total_UPDRS)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", color = "green") +
  ggtitle("RPDE vs total_UPDRS")

# sex vs UPDRS
ggplot(data, aes(x = factor(sex), y = total_UPDRS)) +
  geom_boxplot(fill = "orange") +
  xlab("Sex (0 = Female, 1 = Male)") +
  ggtitle("UPDRS by Sex")

# ----------------------------------------
# Correlation
num_data <- data[, sapply(data, is.numeric)]
cor_matrix <- cor(num_data, use = "complete.obs")
library(corrplot)
corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black")

library(car)
vif(lm(total_UPDRS ~ Jitter_RAP + Shimmer + RPDE + DFA + PPE, data = data)) # VIF check multicollinearity
# Max VIF > 10 or mean VIF >> 1 -> multicollinearity

# =======================================================================================
# Standardized
vars_to_scale <- c("age", "test_time",
                   "Jitter_percent", "Jitter_Abs", "Jitter_RAP", "Jitter_PPQ5", "Jitter_DDP",
                   "Shimmer", "Shimmer_dB", "Shimmer_APQ3", "Shimmer_APQ5", "Shimmer_APQ11", "Shimmer_DDA",
                   "NHR", "HNR", "RPDE", "DFA", "PPE")

# Apply scaling
data[vars_to_scale] <- scale(data[vars_to_scale])

# Model of all variables
model_full <- lmer(
  total_UPDRS ~ age + sex + test_time +
    Jitter_percent + Jitter_Abs + Jitter_RAP + Jitter_PPQ5 + Jitter_DDP +
    Shimmer + Shimmer_dB + Shimmer_APQ3 + Shimmer_APQ5 + Shimmer_APQ11 + Shimmer_DDA +
    NHR + HNR + RPDE + DFA + PPE +
    (1 | subject),
  data = data
)
summary(model_full)

# Test assumption
# 1. Residual plot -- Linearity & Homoscedasticity
plot(fitted(model_full), resid(model_full))
abline(h = 0, col = "red")
# non-random patterns

# 2. Q-Q plot & Shapiro-Wilk test -- Normality
qqnorm(resid(model_full))
qqline(resid(model_full), col = "red")
# Deviations from normality at both ends (heavy tails)

# random effect normality
ranef_model <- ranef(model_full)$subject[, 1]
qqnorm(ranef_model)
qqline(ranef_model, col = "blue")
# not perfectly normal, particularly at the extremes.

hist(resid(model_full), breaks = 50, main = "Histogram of Residuals", xlab = "Residuals")

# =======================================================================================
# Model selection 
# Step 1. LASSO (Handles multicollinearity well)
library(glmnet)
vars_for_lasso <- c("age", "sex", "test_time",
                    "Jitter_percent", "Jitter_Abs", "Jitter_RAP", "Jitter_PPQ5", "Jitter_DDP",
                    "Shimmer", "Shimmer_dB", "Shimmer_APQ3", "Shimmer_APQ5", "Shimmer_APQ11", "Shimmer_DDA",
                    "NHR", "HNR", "RPDE", "DFA", "PPE")
X <- as.matrix(scale(data[, vars_for_lasso]))
y <- data$total_UPDRS

# only consider fix effect for variable selection
cv_lasso <- cv.glmnet(X, y, alpha = 1)  # alpha = 1 -> LASSO
best_lambda <- cv_lasso$lambda.min
lasso_coef <- coef(cv_lasso, s = best_lambda)
lasso_coef

model_lasso_refined <- lmer(total_UPDRS ~ age + sex + test_time +
                              Jitter_Abs + Jitter_RAP + Jitter_PPQ5 + Jitter_DDP +
                              Shimmer_APQ3 + Shimmer_APQ5 + Shimmer_APQ11 + Shimmer_DDA +
                              NHR + HNR + RPDE + DFA + PPE +
                              (1 | subject), data = data)
summary(model_lasso_refined)
# ----------------------------------------------------------
# Step 2. stepwise selection by lmer (based on AIC)
library(lmerTest)  
model_lasso_refined <- lmer(total_UPDRS ~ age + sex + test_time +
                              Jitter_Abs + Jitter_RAP + Jitter_PPQ5 + Jitter_DDP +
                              Shimmer_APQ3 + Shimmer_APQ5 + Shimmer_APQ11 + Shimmer_DDA +
                              NHR + HNR + RPDE + DFA + PPE +
                              (1 | subject), data = data)
stepwise_model <- step(model_lasso_refined)
stepwise_model
model_select <- lmer(total_UPDRS ~ age + test_time + Jitter_PPQ5 + NHR + HNR + (1 | subject), data=data)

# -----------------------------------------------------------
# # Or use cAIC4::stepcAIC()
# library(cAIC4)
# stepwise_model_backward <- stepcAIC(model_full, direction = "backward")
# summary(stepwise_model_backward$merMod)
# # REML: 28109.41 (improved slightly from 28124.71)
# # Best cAIC: 27859.34 (improved from 27869.8)
# # NO variables were removed: This implies that stepcAIC confirmed all variables were contributing under its criterion.

# -------------------------------------------------------------
# Test assumption
# 1. Residual plot -- Linearity & Homoscedasticity
plot(fitted(model_select), resid(model_select), xlab="fitted", ylab="residuals", main="Residual Plot")
abline(h = 0, col = "red")
# non-random patterns

# 2. Q-Q plot & Shapiro-Wilk test -- Normality
qqnorm(resid(model_select))
qqline(resid(model_select), col = "red")
# Deviations from normality at both ends (heavy tails)

# random effect normality
ranef_model <- ranef(model_select)$subject[, 1]
qqnorm(ranef_model)
qqline(ranef_model, col = "blue")
# not perfectly normal, particularly at the extremes.

# Transformation
data$log_total_UPDRS <- log(data$total_UPDRS + 1)
model_log <- lmer(log_total_UPDRS ~ age + test_time + Jitter_PPQ5 + NHR + HNR + (1 | subject), data = data)

plot(fitted(model_log), resid(model_log), xlab="fitted", ylab="residuals", main="Residual Plot")
abline(h = 0, col = "red")
qqnorm(resid(model_log))
qqline(resid(model_log), col = "red")
ranef_model <- ranef(model_log)$subject[, 1]
qqnorm(ranef_model)
qqline(ranef_model, col = "blue")

# =======================================================================================
# Interaction

## Interaction Plot
# test_time's effect on UPDRS change with age
ggplot(data, aes(x = test_time, y = log_total_UPDRS)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~ cut(age, 4)) +  # Binned age
  theme_minimal() +
  ggtitle("Faceted Plot by age Levels")

model_interact_age <- lmer(log_total_UPDRS ~ age * test_time + Jitter_PPQ5 + NHR + HNR + (1 | subject), data = data)
summary(model_interact_age)
AIC(model_select, model_interact_age)  # no age:test_time

# test_time * Jitter_PPQ5: Disease progression linked to voice deterioration
ggplot(data, aes(x = test_time, y = log_total_UPDRS)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~ cut(Jitter_PPQ5, 4)) +  # Binned Jitter_PPQ5
  theme_minimal() +
  ggtitle("Faceted Plot by Jitter_PPQ5 Levels")

model_interact_JP <- lmer(log_total_UPDRS ~ age + test_time * Jitter_PPQ5 + NHR + HNR + (1 | subject), data = data)
summary(model_interact_JP)
AIC(model_select, model_interact_JP)  # remain the interaction

# HNR * NHR: Acoustic features interaction
ggplot(data, aes(x = NHR, y = log_total_UPDRS)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~ cut(HNR, 4)) +  # Binned HNR
  theme_minimal() +
  ggtitle("Faceted Plot by HNR Levels")
model_interact_NH <- lmer(log_total_UPDRS ~ age + test_time * Jitter_PPQ5 + NHR*HNR + (1 | subject), data = data)
summary(model_interact_NH)
AIC(model_select, model_interact_JP, model_interact_NH) # no NHR*HNR

# According to model_interact_JP, drop NHR
model_JP_select <- lmer(log_total_UPDRS ~ age + test_time * Jitter_PPQ5 + HNR + (1 | subject), data = data)
summary(model_JP_select)
AIC(model_select, model_interact_JP, model_JP_select)

# Next, Use model_JP_select
# =======================================================================================
# Random Slopes
# Subject-level Line Plots: If lines have different slopes, random slopes might be needed.
ggplot(data, aes(x = test_time, y = log_total_UPDRS, group = subject)) +
  geom_line(alpha = 0.3) +
  ggtitle("Subject-Specific Trajectories Over Time")

# Plot Residuals by Subject:Large variance in residuals across subjects suggests random slopes could help.
plot(resid(model_JP_select) ~ data$subject)
abline(h = 0, col = "red")

# Plot Fitted Slopes Per Subject: Diverging slopes across subjects = random slopes might improve fit.
fitted_values <- data.frame(fitted = fitted(model_JP_select), subject = data$subject, test_time = data$test_time)
ggplot(fitted_values, aes(x = test_time, y = fitted, group = subject)) +
  geom_line(alpha = 0.4) +
  ggtitle("Fitted Lines for Each Subject")

# Compare AIC
model_slope <- lmer(log_total_UPDRS ~ age + test_time * Jitter_PPQ5 + HNR + (1 + test_time | subject), data = data)
summary(model_slope)
AIC(model_JP_select, model_slope) # model_slope better

# Drop Jitter_PPQ5, add test_time*HNR
model_final <- lmer(log_total_UPDRS ~ age + test_time*HNR + (1 + test_time | subject), data = data)
summary(model_final)
AIC(model_slope, model_final)

# Next, Use model_final
# --------------------------------------------------------
# MSE in original scale
y_hat <- exp(predict(model_final))
y_true <- data$total_UPDRS
mse <- mean((y_true - y_hat)^2)
mse

# =======================================================================================
# plot predicted values for different levels of HNR:
new_data <- expand.grid(test_time = seq(min(data$test_time), max(data$test_time), length.out = 100),
                        HNR = quantile(data$HNR, probs = c(0.25, 0.5, 0.75)),
                        age = mean(data$age),
                        subject = NA)

new_data$pred <- predict(model_final, newdata = new_data, re.form = NA)

ggplot(new_data, aes(x = test_time, y = pred, color = as.factor(round(HNR,2)))) +
  geom_line() +
  labs(title = "Effect of test_time at Different HNR Levels",
       y = "Predicted log(UPDRS)", color = "HNR Level")

# ==========================================================================================
# ==========================================================================================
# If don't use lasso, only use stepwise for model selection
# Step 2. stepwise selection by lmer (based on AIC)
library(lmerTest)  
model_full <- lmer(
  total_UPDRS ~ age + sex + test_time +
    Jitter_percent + Jitter_Abs + Jitter_RAP + Jitter_PPQ5 + Jitter_DDP +
    Shimmer + Shimmer_dB + Shimmer_APQ3 + Shimmer_APQ5 + Shimmer_APQ11 + Shimmer_DDA +
    NHR + HNR + RPDE + DFA + PPE +
    (1 | subject),
  data = data
)
stepwise_model2 <- step(model_full)
stepwise_model2
model_select2 <- lmer(total_UPDRS ~ age + test_time + Jitter_percent + Jitter_Abs + Jitter_RAP + Jitter_PPQ5 + Shimmer + Shimmer_DDA + NHR + HNR + (1 | subject), data=data)
summary(model_select2)
# cor(Shimmer, Shimmer_DDA)=-0.949, drop Shimmer.
model_select3 <- lmer(total_UPDRS ~ age + test_time + Jitter_percent + Jitter_Abs + Jitter_RAP + Jitter_PPQ5 + Shimmer_DDA + NHR + HNR + (1 | subject), data=data)
vif(model_select3)  
# drop Jitter_RAP, Jitter_PPQ5 according to vif
# drop Jitter_Abs, Shimmer according to p-val
model_select4 <- lmer(total_UPDRS ~ age + test_time + Jitter_percent + NHR + HNR + (1 | subject), data = data)
summary(model_select4)
vif(model_select4)

# Test assumption
# 1. Residual plot -- Linearity & Homoscedasticity
plot(fitted(model_select4), resid(model_select4), xlab="fitted value", ylab="residual value")
abline(h = 0, col = "red")
# non-random patterns

# 2. Q-Q plot & Shapiro-Wilk test -- Normality
qqnorm(resid(model_select4))
qqline(resid(model_select4), col = "red")
# Deviations from normality at both ends (heavy tails)

# random effect normality
ranef_model <- ranef(model_select4)$subject[, 1]
qqnorm(ranef_model)
qqline(ranef_model, col = "blue")
# not perfectly normal, particularly at the extremes.

# Transformation
data$log_total_UPDRS <- log(data$total_UPDRS + 1)
model_log2 <- lmer(log_total_UPDRS ~ age + test_time + Jitter_percent + NHR + HNR + (1 | subject), data = data)

plot(fitted(model_log2), resid(model_log2), xlab="fitted value", ylab="residual value")
abline(h = 0, col = "red")
qqnorm(resid(model_log2))
qqline(resid(model_log2), col = "red")
ranef_model2 <- ranef(model_log2)$subject[, 1]
qqnorm(ranef_model2)
qqline(ranef_model2, col = "blue")

# --------------------------------------------------
# Interaction
## Interaction Plot
# test_time's effect on UPDRS change with age
ggplot(data, aes(x = test_time, y = log_total_UPDRS)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~ cut(age, 4)) +  # Binned age
  theme_minimal() +
  ggtitle("Faceted Plot by age Levels")

model_interact_age2 <- lmer(log_total_UPDRS ~ age * test_time + Jitter_percent + NHR + HNR + (1 | subject), data = data)
summary(model_interact_age2)
AIC(model_log2, model_interact_age2)  # no age:test_time

# test_time * Jitter_PPQ5: Disease progression linked to voice deterioration
ggplot(data, aes(x = test_time, y = log_total_UPDRS)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~ cut(Jitter_percent, 4)) +  # Binned Jitter_PPQ5
  theme_minimal() +
  ggtitle("Faceted Plot by Jitter_percent Levels")

model_interact_JP2 <- lmer(log_total_UPDRS ~ age + test_time * Jitter_percent + NHR + HNR + (1 | subject), data = data)
summary(model_interact_JP2)
AIC(model_log2, model_interact_age2, model_interact_JP2)  # remain the interaction

# HNR * NHR: Acoustic features interaction
ggplot(data, aes(x = NHR, y = log_total_UPDRS)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~ cut(HNR, 4)) +  # Binned HNR
  theme_minimal() +
  ggtitle("Faceted Plot by HNR Levels")
model_interact_NH2 <- lmer(log_total_UPDRS ~ age + test_time * Jitter_percent + NHR*HNR + (1 | subject), data = data)
summary(model_interact_NH2)
AIC(model_log2, model_interact_JP2, model_interact_NH2) # no NHR*HNR

# According to model_interact_JP, drop NHR
model_JP_select2 <- lmer(log_total_UPDRS ~ age + test_time * Jitter_percent + HNR + (1 | subject), data = data)
summary(model_JP_select2)
AIC(model_log2, model_interact_JP2, model_JP_select2, model_JP_select)

# test_time * HNR Plots:
ggplot(data, aes(x = test_time, y = log_total_UPDRS)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~ cut(HNR, 4)) +  # Binned HNR
  theme_minimal() +
  ggtitle("Faceted Plot by HNR Levels")

data$HNR_group <- cut(data$HNR, breaks = 3, labels = c("Low", "Mid", "High"))

# Plots 2: group-wise mean
library(dplyr)
summary_df <- data %>%
  group_by(HNR_group, test_time_bin = cut(test_time, 10)) %>%
  summarise(mean_time = mean(test_time),
            mean_updrs = mean(log_total_UPDRS, na.rm = TRUE), .groups = "drop")
ggplot(summary_df, aes(x = mean_time, y = mean_updrs, color = HNR_group)) +
  geom_line(size = 1.2) +
  geom_point() +
  labs(title = "Interaction Plot: test_time × HNR group",
       x = "Test Time", y = "log(UPDRS)", color = "HNR Group") +
  theme_minimal()

# Plots 3
library(ggeffects)
data <- data %>% mutate(HNR_group = ntile(HNR, 3))
model_hnr <- lmer(log_total_UPDRS ~ test_time * HNR_group + (1 | subject), data = data)
plot(ggpredict(model_hnr, terms = c("test_time", "HNR_group")))

model_interact_TH2 <- lmer(log_total_UPDRS ~ age + Jitter_percent + test_time * HNR + (1 | subject), data = data)
summary(model_interact_TH2)
AIC(model_log2, model_interact_JP2, model_JP_select2, model_interact_TH2)
# ----------------------------------------------------------------
# Random Slopes
# Subject-level Line Plots: If lines have different slopes, random slopes might be needed.
ggplot(data, aes(x = test_time, y = log_total_UPDRS, group = subject, col=factor(subject))) +
  geom_line(alpha = 0.4, size=1) +
  ggtitle("Subject-Specific Trajectories Over Time")

# Plot Residuals by Subject:Large variance in residuals across subjects suggests random slopes could help.
plot(resid(model_interact_TH2) ~ data$subject, xlab="Subject", ylab="Residual")
abline(h = 0, col = "red")

# Plot Fitted Slopes Per Subject: Diverging slopes across subjects = random slopes might improve fit.
fitted_values <- data.frame(fitted = fitted(model_interact_TH2), subject = data$subject, test_time = data$test_time)
ggplot(fitted_values, aes(x = test_time, y = fitted, group = subject)) +
  geom_line(alpha = 0.4) +
  ggtitle("Fitted Lines for Each Subject")

# Compare AIC
model_slope2 <- lmer(log_total_UPDRS ~ age + Jitter_percent + test_time *HNR + (1 + test_time | subject), data = data)
summary(model_slope2)
AIC(model_interact_TH2, model_slope2) # model_slope better

# Drop Jitter_percent
model_final2 <- lmer(log_total_UPDRS ~ age + test_time*HNR + (1 + test_time | subject), data = data)
summary(model_final2)
AIC(model_slope2, model_final2)

# adjusted R^2
library(performance)
r2(model_final)

# Next, Use model_final
# --------------------------------------------------------
# MSE in original scale
y_hat <- exp(predict(model_final))
y_true <- data$total_UPDRS
mse <- mean((y_true - y_hat)^2)
mse
# ==========================================================================================
# =======================================================================================
# GAMM + Spline
# install.packages("mgcv")
library(mgcv)

# Fit a GAMM using mgcv with a spline on test_time
gamm_spline <- gamm(log_total_UPDRS ~ age + s(test_time) + HNR, 
                    random = list(subject = ~1 + test_time), 
                    data = data)

# View summaries
summary(gamm_spline$gam)  # Smooth term (test_time)
summary(gamm_spline$lme)  # Random effects (similar to lmer)
# Plot the spline for test_time
plot(gamm_spline$gam, 
     select = 1, 
     shade = TRUE, 
     shade.col = "lightblue",
     se = TRUE, 
     main = "Spline Effect of Test Time",
     xlab = "Test Time", 
     ylab = "Partial Effect")
library(gratia)
gratia::draw(gamm_spline$gam, select = "s(test_time)")

AIC(model_final, gamm_spline$lme)

# MSE in original scale
y_true <- data$total_UPDRS
y_pred_final <- exp(predict(model_final))
y_pred_spline <- exp(predict(gamm_spline$lme))
mse_final <- mean((y_true - y_pred_final)^2)
mse_spline <- mean((y_true - y_pred_spline)^2)

mse_final
mse_spline


# ======================================================================================
# Use training set and testing set to calculate MSE in original scale
# Split subjects into training and testing
data$subject <- factor(data$subject)  

test_data <- data %>%
  group_by(subject) %>%
  slice_tail(n = 1) %>%
  ungroup()
train_data <- data %>%
  group_by(subject) %>%
  filter(row_number() < n()) %>%
  ungroup()
model_final2 <- lmer(log_total_UPDRS ~ age + test_time*HNR + (1 + test_time | subject), data = train_data)

y_pred <- exp(predict(model_final2, newdata = test_data, allow.new.levels = TRUE))
y_true <- test_data$total_UPDRS

mse <- mean((y_true - y_pred)^2)
print(paste("Test MSE (original scale):", round(mse, 4)))

mae_lmm <- mean(abs(test_data$total_UPDRS - y_pred))

# ----------------------------------------------------
# mse of spline
gamm_spline2 <- gamm(log_total_UPDRS ~ age + s(test_time) + HNR,
                     random = list(subject = ~1 + test_time),
                     data = train_data)

pred_fixed <- predict(gamm_spline2$gam, newdata = test_data)


ranef_vals <- ranef(gamm_spline2$lme)$subject
subject_ids <- as.character(test_data$subject)
subject_keys <- paste0("1/", subject_ids)  


re_effect <- mapply(function(s, t) {
  if (s %in% rownames(ranef_vals)) {
    b0 <- ranef_vals[s, "(Intercept)"]
    b1 <- ranef_vals[s, "test_time"]
    return(b0 + b1 * t)
  } else {
    return(0)  
  }
}, s = subject_keys, t = test_data$test_time)


log_pred_gamm_full <- pred_fixed + re_effect
y_pred <- exp(log_pred_gamm_full)
y_true <- test_data$total_UPDRS

mse_gamm_full <- mean((y_true - y_pred)^2)
print(paste("Test MSE with full GAMM (FE + RE):", round(mse_gamm_full, 2)))

mae_gamm <- mean(abs(y_true - y_pred))
