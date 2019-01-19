# # -----------------------------------------------------------------------
# Nick Seedorff
# Machine Learning Final Project
# April 12th 2018
# Description: Final project for course in machine learning. Used a
# vaiety of methods to classify (predict) pass/fail outcomes of high 
# school students. Final model chosen using k-fold CV.
# # -----------------------------------------------------------------------

# Libraries ---------------------------------------------------------------
library("caret")
library("recipes")
library("MASS")
library("kernlab")
library("gbm")
library("plyr")
library("dplyr")
library("gam")
library("randomForest")
library("ipred")
library("e1071")
library("rpart")
library("gridExtra")

# Data --------------------------------------------------------------------
data <- as.data.frame(read.csv("Class.csv")[,c(-1, -32, -33)])
summary(data)
ord <- c(7, 8, 13, 14, 15, 24, 25, 26, 27, 28, 29)
num <- c(3, 15, 30)
Pass <- as.factor(data$G3 >=10)
data$Medu[which(data$Medu==0)] <- 1
data$Fedu[which(data$Fedu==0)] <- 1

# test and training sets
set.seed(24)
full.dat <- cbind(Pass, data[,1:30])
trainIndex <- createDataPartition(data$G3, p = 0.7, list = F, times=1)
train.data <- full.dat[trainIndex,]
test.data <- full.dat[-trainIndex,]

# Descriptive statistics --------------------------------------------------
sumFunc <- function(x) c(mean(x), sd(x), min(x), max(x))
sum.stat <- round(t(apply(data[,num], 2, sumFunc)),3)
colnames(sum.stat) <- c("Mean", "SD", "Min", "Max")
write.csv(sum.stat, "summaryStats.csv")

#binary variable plots
bin.var <- c("school", "sex", "address", "famsize", "Pstatus", "schoolsup",
             "famsup", "paid", "activities", "nursery", "higher", "internet",
             "romantic")
bin.list <- vector("list", length(bin.var))

for(i in 1:length(bin.var)){
  bin.list[[i]] <- ggplot(data, aes_string(x = bin.var[i], fill=bin.var[i])) +
    geom_bar(aes(y = (..count..)/sum(..count..))) + 
    guides(fill=FALSE) +
    ylim(0,1) +
    theme(text = element_text(size=16), axis.title.y=element_blank())
}

dev.new(width=6, height=4)
grid.arrange(bin.list[[1]], bin.list[[2]], bin.list[[3]], bin.list[[4]],
             bin.list[[5]], bin.list[[6]], bin.list[[7]], bin.list[[8]],
             bin.list[[9]], bin.list[[10]], bin.list[[11]], bin.list[[12]],
             bin.list[[13]], nrow=4)
#end binary variables

#multicategory plots
mult.var <- c("Mjob", "Fjob", "reason", "guardian")
mult.list <- vector("list", length(mult.var))

for(i in 1:length(mult.var)){
  mult.list[[i]] <- ggplot(data, aes_string(x = mult.var[i], fill=mult.var[i])) +
    geom_bar(aes(y = (..count..)/sum(..count..))) + 
    guides(fill=FALSE) +
    ylim(0,1) +
    theme(text = element_text(size=16), axis.title.y=element_blank(),
          axis.text.x = element_text(angle=90, hjust=0.5))
}

dev.new(width=6, height=4)
grid.arrange(mult.list[[1]], mult.list[[2]], mult.list[[3]], mult.list[[4]], nrow=2)
#end multicategory variables

#ordinal plots
ord.var <- c("Medu", "Fedu", "traveltime", "studytime", "famrel", "freetime", "goout", "Dalc",
             "Walc", "health")
ord.list <- vector("list", length(ord.var))

for(i in 1:length(ord.var)){
  ord.list[[i]] <- ggplot(data, aes_string(x = ord.var[i], fill=ord.var[i])) +
    geom_bar(aes(y = (..count..)/sum(..count..))) + 
    guides(fill=FALSE) +
    ylim(0,1) +
    theme(text = element_text(size=16), axis.title.y=element_blank())
}

dev.new(width=6, height=4)
grid.arrange(ord.list[[1]], ord.list[[2]], ord.list[[3]], ord.list[[4]],
             ord.list[[5]], ord.list[[6]], ord.list[[7]], ord.list[[8]],
             ord.list[[9]], ord.list[[10]], nrow=3)
#end ordinal variables

# Resampling ------------------------------------------------------------------
trainControl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3
)

#sbfcontrol
filterControl <- sbfControl(
  functions = caretSBF,
  method = "repeatedcv",
  number = 10,
  repeats = 3
)

# Recipe ------------------------------------------------------------------
train_recipe <- recipe(Pass ~ ., data = train.data) %>%
  check_missing(Pass) %>%
  step_nzv(all_predictors()) %>%
  step_num2factor(all_numeric(), - age, -failures, -absences, ordered = TRUE) %>%
  step_other(all_nominal(), -all_outcomes(), threshold = 0.05, other = "otherValues") %>%
  step_center(all_numeric()) %>%
  step_scale(all_numeric()) 

#data for models with filtering
train.data2 <- juice(prep(train_recipe, retain=T))

# Tree based methods ------------------------------------------------------

#Single tree
set.seed(24)
treegrid <- expand.grid(cp = seq(0.02, 0.2, 0.02)) 
fit.tree <- train(train_recipe, data = train.data, method = "rpart", trControl = trainControl,
                  tuneGrid= treegrid) 
fit.tree$results
dev.new(width=6, height=4)
ggplot(varImp(fit.tree, cuts=5), mapping = NULL) + 
  ggtitle("CART Variable Importance") + 
  theme(text = element_text(size=18))

#Bagging
set.seed(24)
fit.bag <- train(train_recipe, data = train.data, method = "treebag", trControl = trainControl) 
fit.bag$results
ggplot(varImp(fit.bag), mapping = NULL) + 
  ggtitle("Bagged Trees Variable Importance") + 
  theme(text = element_text(size=18))

#Random Forest
set.seed(24)
rfgrid <- expand.grid(mtry = seq(5, 29, 2))
fit.rf <- train(train_recipe, data = train.data, method = "rf", trControl = trainControl,
                tuneGrid = rfgrid) 
fit.rf$results
ggplot(varImp(fit.rf), mapping = NULL) + 
  ggtitle("Random Forest Variable Importance") + 
  theme(text = element_text(size=18))

#GBM
set.seed(24)
gbmgrid <- expand.grid(shrinkage = seq(0.01, .1, 0.02), n.trees = c(50, 100, 150),
                       interaction.depth = c(1, 2, 3), n.minobsinnode= 10)
gbm.res <- as.data.frame(matrix(NA, nrow = dim(gbmgrid)[1], ncol = 8))

for(i in 1:dim(gbmgrid)[1]){
  (fit.gbm <- train(train_recipe, data=train.data, trControl = trainControl, method="gbm",
                    tuneGrid = gbmgrid[i,]))
  gbm.res[i,] <- fit.gbm$results
  
  #keep best model
  if (i == 1){
    keep.gbm <- fit.gbm
  } else if (fit.gbm$results[5] > keep.gbm$results[5]){
    keep.gbm <- fit.gbm
  }
}

which(gbm.res$V5 == max(gbm.res$V5))

dev.new(width=6, height=4)
ggplot(varImp(keep.gbm), mapping = NULL) + 
  ggtitle("GBM Variable Importance") + 
  theme(text = element_text(size=18))

#write.csv(gbm.res, file="gbmres.csv")
#write.csv(gbm.var, file="gbmvar.csv")
#write.csv(gbmgrid, file="gbmgrid.csv") 

# Other Models ------------------------------------------------------------

# Recipe2 ------------------------------------------------------------------
# same recipe, but manual dummy variables
train_recipe2 <- recipe(Pass ~ ., data = train.data) %>%
  check_missing(Pass) %>%
  step_nzv(all_predictors()) %>%
  step_num2factor(all_numeric(), - age, -failures, -absences, ordered = TRUE) %>%
  step_other(all_nominal(), -all_outcomes(), threshold = 0.05, other = "otherValues") %>%
  step_center(all_numeric()) %>%
  step_scale(all_numeric()) %>%
  step_dummy(all_nominal(), -all_outcomes())

#data for models with filtering
train.data3 <- juice(prep(train_recipe2, retain=T))

# Neural Network
set.seed(24)
nnetgrid <- expand.grid(size = seq(5, 20, 5), decay = seq(0, 0.20,0.05))
nnet.res <- as.data.frame(matrix(NA, nrow = dim(nnetgrid)[1], ncol = 4))

for(i in 1:dim(nnetgrid)[1]){
  fit.nnet <- sbf(Pass ~., data = train.data3, method = "nnet", sbfControl = filterControl,
                  tuneGrid= nnetgrid[i,])
  
  nnet.res[i,] <- fit.nnet$results
  
  #keep best model
  if (i == 1){
    keep.nonlinear <- fit.nnet
  } else if (fit.nnet$results[1] > keep.nonlinear$results[1]){
    keep.nonlinear <- fit.nnet
  }
}

ggplot(varImp(fit.nnet$fit), mapping = NULL) + 
  ggtitle("Neural Network Variable Importance") +
  theme(text = element_text(size=18))

write.csv(nnet.res, file="nnetres.csv")
write.csv(nnetgrid, file="nnetgrid.csv")

#logistic regression
set.seed(24)
fit.log <- sbf(Pass ~., data = train.data3, method = "glm", family = "binomial", sbfControl = filterControl)
fit.log$results
ggplot(varImp(fit.log$fit), mapping = NULL) + 
  ggtitle("Logistic Regression Variable Importance") + 
  theme(text = element_text(size=18))

#lda
set.seed(24)
fit.lda <- sbf(Pass ~., data = train.data3, method = "lda", sbfControl = filterControl)
fit.lda$results

ggplot(varImp(fit.lda$fit), mapping = NULL) + 
  ggtitle("LDA Variable Importance") + 
  theme(text = element_text(size=18))

# Resampling plots --------------------------------------------------------
resamp <- resamples(list(CART = fit.tree, Bagging = fit.bag, RandomF = fit.rf, GBM = keep.gbm, 
                         NueralNet = keep.nonlinear, Logistic = fit.log, LDA = fit.lda))

summary(resamp)
bwplot(resamp, scales=list(y=list(cex=1.5), x=list(cex=1)))


# Predictive performance on the test set ----------------------------------
final.dat <- bake(prep(train_recipe), test.data)
sum(predict(keep.gbm, final.dat) == final.dat$Pass)/length(final.dat$Pass)
