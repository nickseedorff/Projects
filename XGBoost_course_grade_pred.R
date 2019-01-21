#######################################################################################
#Nick Seedorff
#1/18/2019
#Description: Predictive modeling using xgboost at Weeks 5 and 10
#######################################################################################

# Packages
library(xgboost)
library(MachineShop)
library(dplyr)
source("X:\\_NickS\\Functions\\pre_analysis_func.R") #Own functions

# Data
data <- read.csv("X:\\_NickS\\Analysis\\Spring2019\\Course_predictive_modeling\\modeling_data.csv")

# Split
set.seed(24)
trainIndex <- sample(nrow(data), nrow(data) * 0.7)
train.data <- data[trainIndex,]
test.data <- data[-trainIndex,]

# Prior grade Distributions
grade_dist <- read.csv("X:\\_NickS\\Analysis\\Spring2019\\Course_predictive_modeling\\grade_dist.csv")
grade_dist <- grade_dist %>%
  mutate(cumsum = cumsum(DISTRIBUTION_PERCENTAGE)) %>%
  mutate(cutoff = cumsum/100 * dim(test.data)[1]) # Cutoffs to classify grade predictions with

# Week 5 Model ------------------------------------------------------------

# Week 5 features
features5 <- c("FINAL_SCORE", 
              "Exams_05", "Discussion_Participation_03", "Discussion_Participation_04", "Discussion_Participation_05",
              "MC_Homework_03", "MC_Homework_04", "MC_Homework_05",
              "ICON_Quizzes_.IQs._01", "ICON_Quizzes_.IQs._02", "ICON_Quizzes_.IQs._03", "ICON_Quizzes_.IQs._04", "ICON_Quizzes_.IQs._05")

# Week 5 data
week5.data <- train.data[, features5]
week5.test <- test.data[, features5]

# Model formula
mod.form <- FINAL_SCORE ~ .

## Tune tree depth
results.list <- list() # List to store results
results.vec <- NA # Vector to store mean RMSE

for(i in 1:8){
  gbmfit <- resample(mod.form, data = week5.data, 
                   model = XGBModel(nrounds = 200, params = list(max_depth = i)), 
                   control = CVControl(fold = 10, repeats = 1, seed=24))
  results.list[[i]] <- summary(gbmfit) # Store all performance metrics
  results.vec[i] <- summary(gbmfit)[2, 1] #Store mean RMSE
}

## Tune learning parameter
eta.vec <- seq(0.05, 0.5, by = 0.05) # Learning parameter values to loop through
results.list2 <- list() # List to store results
results.vec2 <- NA # Vector to store mean RMSE

for(i in 1:length(eta.vec)){
  gbmfit <- resample(mod.form, data = week5.data, 
                     model = XGBModel(nrounds = 250, params = list(eta = eta.vec[i], max.depth = min(which(results.vec == min(results.vec))))), 
                     control = CVControl(fold = 10, repeats = 2, seed=24)) #Fit model using chosen max.depth from previous fitting
  results.list2[[i]] <- summary(gbmfit) # Store all performance metrics
  results.vec2[i] <- summary(gbmfit)[2, 1] # store mean RMSE
}

## Define final tuning parameters and fit final model
final.depth <- min(which(results.vec == min(results.vec))) #Min chooses simplest model if two are equal
final.eta <- eta.vec[min(which(results.vec2 == min(results.vec2)))] 
final.fit <- fit(mod.form, data = week5.data, 
                   model = XGBModel(nrounds = 250, params = list(max.depth = final.depth, eta = final.eta)))

#Variable importance
varimp(final.fit)

# Out of sample performance metrics ---------------------------------------

# Preprocessing of data for performance metrics
predictions_5 <- data.frame(test.data$UNIVERSITY_ID, predict(final.fit, newdata = week5.test)) 
colnames(predictions_5) <- c("UNIVERSITY_ID", "PRED_SCORE")
predictions_5 <- predictions_5[order(predictions_5$PRED_SCORE, decreasing = TRUE), ] #Order obs by pred score
predictions_5$ORDER <- seq(1, length(predictions_5$UNIVERSITY_ID) , 1) # Give the order for use in functions

# Use prior functions to classify data
predictions_5 <- Pred_Cutoff_Grades(predictions_5) # Classifiy grade predictions according to distribution
predictions_5 <- Num_Grades_Func(predictions_5) # Get numeric grade values to compare with
colnames(predictions_5)[4:5] <- c("GRADE_PRED_VALUE", "GRADE_PRED_VALUE_NUMERIC") 

#Merge with test data
compare_5 <- left_join(predictions_5, test.data[, c("UNIVERSITY_ID", "GRADE_VALUE", "GRADE_VALUE_NUMERIC")], by = "UNIVERSITY_ID")

#Set grade value to of F to 0.33 for +/- calculations
compare_5$GRADE_PRED_VALUE_NUMERIC[compare_5$GRADE_PRED_VALUE_NUMERIC==0] = 0.33
compare_5$GRADE_VALUE_NUMERIC[compare_5$GRADE_VALUE_NUMERIC==0] = 0.33

compare_5$CORRECT <- as.numeric(compare_5$GRADE_PRED_VALUE_NUMERIC == compare_5$GRADE_VALUE_NUMERIC)
compare_5$CORRECT2 <- as.numeric(compare_5$GRADE_PRED_VALUE_NUMERIC - 0.5 < compare_5$GRADE_VALUE_NUMERIC & 
                                   compare_5$GRADE_VALUE_NUMERIC < compare_5$GRADE_PRED_VALUE_NUMERIC + 0.5)

## Performance metrics 
metrics <- performance(week5.test$FINAL_SCORE, predict(final.fit, newdata = week5.test)) #Out of sample peformance metrics
metrics_5 <- data.frame(t(metrics), ACC = sum(compare_5$CORRECT)/length(compare_5$CORRECT), 
                        ACC_PLUS_MINUS =  sum(compare_5$CORRECT2)/length(compare_5$CORRECT))

# Week 10 Model ------------------------------------------------------------

# Week 10 features
features10 <- c("FINAL_SCORE", 
               "Exams_05", "Exams_06","Exams_09","Exams_10",
               "Discussion_Participation_03", "Discussion_Participation_04", "Discussion_Participation_05", "Discussion_Participation_07",
               "Discussion_Participation_08", "Discussion_Participation_09", "Discussion_Participation_10",
               "MC_Homework_03", "MC_Homework_04", "MC_Homework_05", "MC_Homework_06", "MC_Homework_07", "MC_Homework_08",
               "MC_Homework_09", "MC_Homework_10",
               "ICON_Quizzes_.IQs._01", "ICON_Quizzes_.IQs._02", "ICON_Quizzes_.IQs._03", "ICON_Quizzes_.IQs._04", "ICON_Quizzes_.IQs._05",
               "ICON_Quizzes_.IQs._06", "ICON_Quizzes_.IQs._07", "ICON_Quizzes_.IQs._08", "ICON_Quizzes_.IQs._09", "ICON_Quizzes_.IQs._10")

# Week 5 data
week10.data <- train.data[, features10]
week10.test <- test.data[, features10]

# Model formula
mod.form <- FINAL_SCORE ~ .

## Tune tree depth
results.list <- list() # List to store results
results.vec <- NA # Vector to store mean RMSE

for(i in 1:8){
  gbmfit <- resample(mod.form, data = week10.data, 
                     model = XGBModel(nrounds = 200, params = list(max_depth = i)), 
                     control = CVControl(fold = 10, repeats = 1, seed=24))
  results.list[[i]] <- summary(gbmfit) # Store all performance metrics
  results.vec[i] <- summary(gbmfit)[2, 1] # Store mean RMSE
}

## Tune learning parameter
eta.vec <- seq(0.05, 0.5, by = 0.05) # Learning parameter values to loop through
results.list2 <- list() # List to store results
results.vec2 <- NA # Vector to store mean RMSE

for(i in 1:length(eta.vec)){
  gbmfit <- resample(mod.form, data = week10.data, 
                     model = XGBModel(nrounds = 250, params = list(eta = eta.vec[i], max.depth = min(which(results.vec == min(results.vec))))), 
                     control = CVControl(fold = 10, repeats = 2, seed=24)) #Fit model using chosen max.depth from previous fitting
  results.list2[[i]] <- summary(gbmfit) # Store all performance metrics
  results.vec2[i] <- summary(gbmfit)[2, 1] # Store mean RMSE
}

## Fit chosen model
final.depth <- min(which(results.vec == min(results.vec))) #Min chooses simplest model if two are equal
final.eta <- eta.vec[min(which(results.vec2 == min(results.vec2)))]

final.fit <- fit(mod.form, data = week10.data, 
                 model = XGBModel(nrounds = 250, params = list(max.depth = final.depth, eta = final.eta)))
varimp(final.fit)


# Out of sample performance metrics ---------------------------------------

# Preprocessing of data for performance metrics
predictions_10 <- data.frame(test.data$UNIVERSITY_ID, predict(final.fit, newdata = week10.test)) 
colnames(predictions_10) <- c("UNIVERSITY_ID", "PRED_SCORE")
predictions_10 <- predictions_10[order(predictions_10$PRED_SCORE, decreasing = TRUE), ] #Order obs by pred score
predictions_10$ORDER <- seq(1, length(predictions_10$UNIVERSITY_ID) , 1) # Give the order for use in functions

# Use prior functions to classify data
predictions_10 <- Pred_Cutoff_Grades(predictions_10) # Classifiy grade predictions according to distribution
predictions_10 <- Num_Grades_Func(predictions_10) # Get numeric grade values to compare with
colnames(predictions_10)[4:5] <- c("GRADE_PRED_VALUE", "GRADE_PRED_VALUE_NUMERIC") 

#Merge with test data
compare_10 <- left_join(predictions_10, test.data[, c("UNIVERSITY_ID", "GRADE_VALUE", "GRADE_VALUE_NUMERIC")], by = "UNIVERSITY_ID")

#Set grade value to of F to 0.33 for +/- calculations
compare_10$GRADE_PRED_VALUE_NUMERIC[compare_10$GRADE_PRED_VALUE_NUMERIC==0] = 0.33
compare_10$GRADE_VALUE_NUMERIC[compare_10$GRADE_VALUE_NUMERIC==0] = 0.33

compare_10$CORRECT <- as.numeric(compare_10$GRADE_PRED_VALUE_NUMERIC == compare_10$GRADE_VALUE_NUMERIC)
compare_10$CORRECT2 <- as.numeric(compare_10$GRADE_PRED_VALUE_NUMERIC - 0.5 < compare_10$GRADE_VALUE_NUMERIC & 
                                   compare_10$GRADE_VALUE_NUMERIC < compare_10$GRADE_PRED_VALUE_NUMERIC + 0.5)

## Performance metrics 
metrics <- performance(week10.test$FINAL_SCORE, predict(final.fit, newdata = week10.test)) #Out of sample peformance metrics
metrics_10 <- data.frame(t(metrics), ACC = sum(compare_10$CORRECT)/length(compare_10$CORRECT), 
                        ACC_PLUS_MINUS =  sum(compare_10$CORRECT2)/length(compare_10$CORRECT))

# Performance metrics on test set -----------------------------------------

final.metric <- rbind(metrics_5, metrics_10)
