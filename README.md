# Employee Attrition Analysis with SVP & Random Forest

**SETUP Packages**
library(dplyr)         # For data manipulation
library(ggplot2)       # For data visualization
library(caret)         # For machine learning workflow support
library(rpart)         # For decision trees
library(e1071)         # For SVM
library(randomForest)  # For Random Forest
library(ada)           # For AdaBoost
library(ROCR)          # For AUC/ROC calculations
library(pdp)           # For Partial Dependence Plots (PDP) - Model Interpretation
library(lime)          # For LIME - Model Interpretation

**1. DATA LOADING AND EXPLORATION**
MYdataset <- read.csv("Attrition_Prediction.csv")
str(MYdataset)
summary(MYdataset)

**2. DATA PREPROCESSING AND CLEANING**
vars_to_drop <- c("Over18", "EmployeeCount", "StandardHours", "EmployeeNumber")
MYdataset <- MYdataset %>% select(-all_of(vars_to_drop))
conv_fact <- c("Education", "EnvironmentSatisfaction", "JobInvolvement", "JobLevel", "JobSatisfaction", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "TrainingTimesLastYear", "WorkLifeBalance")
MYdataset[, conv_fact] <- lapply(MYdataset[, conv_fact], as.factor)
MYdataset$Attrition <- as.factor(MYdataset$Attrition)

**3. EXPLORATORY DATA VISUALIZATION (EDA)**
**4. DATA SPLITTING & ADVANCED PREPROCESSING**
set.seed(123) 
trainIndex <- createDataPartition(MYdataset$Attrition, p = .75, list = FALSE)
ModelData <- MYdataset[trainIndex, ]
ValidateData <- MYdataset[-trainIndex, ]
Subject <- c("OverTime", "MonthlyIncome", "TotalWorkingYears", "HourlyRate", "JobRole", "Age")
Objective <- "Attrition"
TrainingData <- ModelData[, c(Subject, Objective)]
ValidationData <- ValidateData[, c(Subject, Objective)]

**5. Advanced Preprocessing: Scaling Numeric Variables** 
preProcValues <- preProcess(TrainingData, method = c("center", "scale"))
TrainingData_scaled <- predict(preProcValues, TrainingData)
ValidationData_scaled <- predict(preProcValues, ValidationData)
str(TrainingData_scaled)

**6. Checking for STATISTICAL SIGNIFICANCE of Predictors**
logit_model <- glm(Attrition ~ ., data = TrainingData, family = binomial(link = "logit"))
print(summary(logit_model))

**7. PREDICTIVE MODELING & EVALUATION (Using SCALED Data)**
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

**--- MODEL 1: Support Vector Machine (SVM) ---**
print("--- Training SVM on SCALED data ---")
tune_results <- tune(svm, Attrition ~ ., data = TrainingData_scaled, kernel = 'radial',
                     ranges = list(cost = c(1, 5, 10), gamma = c(0.1, 0.5, 1)))
model_SVM <- svm(Attrition ~ ., data = TrainingData_scaled, cost = tune_results$best.parameters$cost, gamma = tune_results$best.parameters$gamma, probability = TRUE)
predict_SVM <- predict(model_SVM, ValidationData_scaled)
result_SVM <- confusionMatrix(data = predict_SVM, reference = ValidationData_scaled$Attrition, positive = "Yes")
auc_SVM <- calculate_auc(model_SVM, ValidationData_scaled)
print(paste("SVM AUC:", round(auc_SVM, 4)))

**--- MODEL 2: Random Forest ---**
# (Note: Tree-based models like Random Forest are not sensitive to scaling, 
# but we use the scaled data for consistency in our workflow).
print("--- Training Random Forest on SCALED data ---")
# [UPDATE] Using scaled data. We will use this model for interpretation later.
model_RF <- randomForest(Attrition ~ ., data = TrainingData_scaled, ntree = 500, mtry = 2, importance = TRUE)
predict_RF <- predict(model_RF, ValidationData_scaled)
result_RF <- confusionMatrix(data = pred_RF, reference = ValidationData_scaled$Attrition, positive = "Yes")
auc_RF <- calculate_auc(model_RF, ValidationData_scaled)
print(paste("Random Forest AUC:", round(auc_RF, 4)))
# ===================================================================
