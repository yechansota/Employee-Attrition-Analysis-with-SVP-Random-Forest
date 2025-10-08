# 🧠 Employee Attrition Analysis  
### Using Support Vector Machine (SVM) & Random Forest

This project analyzes **employee attrition** using multiple machine learning models (SVM & Random Forest).  
It also demonstrates data preprocessing, model tuning, evaluation (AUC/ROC), and interpretability techniques (PDP, LIME).

---

## 📦 1. Setup — Load Required Packages
library(dplyr)         # Data manipulation
library(ggplot2)       # Visualization
library(caret)         # ML workflow utilities
library(rpart)         # Decision trees
library(e1071)         # SVM
library(randomForest)  # Random Forest
library(ada)           # AdaBoost
library(ROCR)          # ROC/AUC performance
library(pdp)           # Partial Dependence Plot (Model interpretation)
library(lime)          # LIME (Local Interpretable Model Explanation)

MYdataset <- read.csv("Attrition_Prediction.csv")
str(MYdataset)
summary(MYdataset)

# Drop irrelevant variables
vars_to_drop <- c("Over18", "EmployeeCount", "StandardHours", "EmployeeNumber")
MYdataset <- MYdataset %>% select(-all_of(vars_to_drop))

# Convert selected columns to factors
conv_fact <- c("Education", "EnvironmentSatisfaction", "JobInvolvement", "JobLevel",
               "JobSatisfaction", "PerformanceRating", "RelationshipSatisfaction",
               "StockOptionLevel", "TrainingTimesLastYear", "WorkLifeBalance")
MYdataset[, conv_fact] <- lapply(MYdataset[, conv_fact], as.factor)

# Ensure target variable is a factor
MYdataset$Attrition <- as.factor(MYdataset$Attrition)

set.seed(123)
trainIndex <- createDataPartition(MYdataset$Attrition, p = 0.75, list = FALSE)
ModelData <- MYdataset[trainIndex, ]
ValidateData <- MYdataset[-trainIndex, ]

Subject <- c("OverTime", "MonthlyIncome", "TotalWorkingYears", "HourlyRate", "JobRole", "Age")
Objective <- "Attrition"

TrainingData <- ModelData[, c(Subject, Objective)]
ValidationData <- ValidateData[, c(Subject, Objective)]

preProcValues <- preProcess(TrainingData, method = c("center", "scale"))
TrainingData_scaled <- predict(preProcValues, TrainingData)
ValidationData_scaled <- predict(preProcValues, ValidationData)

str(TrainingData_scaled)

logit_model <- glm(Attrition ~ ., data = TrainingData, family = binomial(link = "logit"))
summary(logit_model)

calculate_auc <- function(model, validation_data) {
  if (inherits(model, "svm")) {
    prob_preds <- predict(model, validation_data, probability = TRUE)
    prob_yes <- attr(prob_preds, "probabilities")[, "Yes"]
  } else if (inherits(model, "randomForest")) {
    prob_preds <- predict(model, validation_data, type = "prob")
    prob_yes <- prob_preds[, "Yes"]
  } else {
    prob_preds <- predict(model, validation_data, type = "prob")
    prob_yes <- prob_preds[, 2]
  }
  pred_obj <- prediction(prob_yes, validation_data$Attrition)
  auc_obj <- performance(pred_obj, measure = "auc")
  return(auc_obj@y.values[[1]])
}

print("--- Training SVM on SCALED data ---")

# Hyperparameter tuning
tune_results <- tune(
  svm, Attrition ~ ., data = TrainingData_scaled, kernel = 'radial',
  ranges = list(cost = c(1, 5, 10), gamma = c(0.1, 0.5, 1))
)

# Fit the model with best parameters
model_SVM <- svm(
  Attrition ~ ., data = TrainingData_scaled,
  cost = tune_results$best.parameters$cost,
  gamma = tune_results$best.parameters$gamma,
  probability = TRUE
)

# Prediction & Evaluation
predict_SVM <- predict(model_SVM, ValidationData_scaled)
result_SVM <- confusionMatrix(data = predict_SVM, reference = ValidationData_scaled$Attrition, positive = "Yes")
auc_SVM <- calculate_auc(model_SVM, ValidationData_scaled)
print(paste("✅ SVM AUC:", round(auc_SVM, 4)))

print("--- Training Random Forest on SCALED data ---")

model_RF <- randomForest(
  Attrition ~ ., data = TrainingData_scaled,
  ntree = 500, mtry = 2, importance = TRUE
)

predict_RF <- predict(model_RF, ValidationData_scaled)
result_RF <- confusionMatrix(data = predict_RF, reference = ValidationData_scaled$Attrition, positive = "Yes")
auc_RF <- calculate_auc(model_RF, ValidationData_scaled)

print(paste("✅ Random Forest AUC:", round(auc_RF, 4)))


# Partial Dependence Plot
pdp::partial(model_RF, pred.var = "MonthlyIncome", plot = TRUE)

# LIME Interpretation
explainer <- lime(TrainingData_scaled[, -ncol(TrainingData_scaled)], model_RF)
explanation <- explain(ValidationData_scaled[1:5, -ncol(ValidationData_scaled)], explainer, n_labels = 1, n_features = 3)
plot_features(explanation)

