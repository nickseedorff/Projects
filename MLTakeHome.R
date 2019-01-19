# # -----------------------------------------------------------------------
# Nick Seedorff
# July 23rd
# ML Take home
# Description: Machline learning algorithms to classify binary 
# diagnosis of diseased individuals. Part of comprehensive exam.
# # -----------------------------------------------------------------------

#Working Directory

#Packages
library(elasticnet)
library(ggplot2)
library(gridExtra)
library(gplots)
library(caret)
library(recipes)
library(gbm)
library(pROC)

#Data
data <- lung
data.models <- data[, -which(names(data)=="StudyID")]
data.models$Race <- as.factor(data.models$Race == "Caucasian")
data.models$BinaryDiagnosis <- as.factor(data.models$BinaryDiagnosis)

#Split data
set.seed(24)
trainIndex <- createDataPartition(data$BinaryDiagnosis, p = 0.7, list = F, times=1)
train.data <- data.models[trainIndex,]
test.data <- data.models[-trainIndex,]

# Descriptive and Summary Stats -------------------------------------------
#Outcome
sum(data$BinaryDiagnosis)/length(data$BinaryDiagnosis)

#Continuous Features
num <- which(names(data) %in% c("Age", "Pack_Years"))
sumFunc <- function(x) c(mean(x), sd(x), median(x), min(x), max(x))
sum.stat <- round(t(apply(data[,num], 2, sumFunc)),3)
colnames(sum.stat) <- c("Mean", "SD", "Med", "Min", "Max")
sum.stat

sum.stat <- round(t(apply(data[data$BinaryDiagnosis==1,num], 2, sumFunc)),3) #diseased
sum.stat <- round(t(apply(data[data$BinaryDiagnosis==0,num], 2, sumFunc)),3) #benign

#Nominal variable plots
bin.var <- c("Lobe","Race", "Gender")
bin.list <- vector("list", length(bin.var))

data2 <- data
for(i in 1:length(bin.var)){
  dframe <- aggregate(as.formula(paste("BinaryDiagnosis ~ ", bin.var[i])), FUN = mean, data=data2)
  bin.list[[i]] <- ggplot(data, aes_string(x = bin.var[i], fill=bin.var[i])) +
    geom_bar(aes(y = (..count..)/sum(..count..))) + 
    guides(fill=FALSE) +
    ylim(0,1) +
    theme(text = element_text(size=16), axis.title.y=element_blank()) + 
    geom_point(data=dframe, aes_string(x = bin.var[i], y = "BinaryDiagnosis"), size = 4) +
    geom_line(data = dframe, aes(group=1, y=BinaryDiagnosis), size = 1)
}

dev.new(width=4, height=4)
grid.arrange(bin.list[[1]], bin.list[[2]], bin.list[[3]], nrow=3)

#Heat map of remaining continous features
###heatmap
#col2 <- rev(heat.colors(64))
#dat.heatmap <- scale(data[, which(names(data)=="NodeFeat1"):which(names(data)=="RecistMaxDiam")], center=T, scale=T)
#dev.new(width=6, height=6)
#heatmap.2(t(dat.heatmap), col=col2, trace="none",
#          margin=c(1,5), labCol="")

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
train_recipe <- recipe(BinaryDiagnosis ~ ., data = train.data) %>%
  check_missing(BinaryDiagnosis) %>%
  step_nzv(all_predictors()) %>%
  step_other(all_nominal(), -all_outcomes(), threshold = 0.05, other = "otherValues") %>%
  step_corr(all_numeric(), -all_outcomes(), threshold = 0.9) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) 

# Modeling ----------------------------------------------------------------

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
  theme(text = element_text(size=13))
#write.csv(gbm.res, file="gbmres.csv")
#write.csv(gbmgrid, file="gbmgrid.csv") 

# Recipe2 ------------------------------------------------------------------
# same recipe, but manual dummy variables
train_recipe2 <- recipe(BinaryDiagnosis ~ ., data = train.data) %>%
  check_missing(BinaryDiagnosis) %>%
  step_nzv(all_predictors()) %>%
  step_other(all_nominal(), -all_outcomes(), threshold = 0.05, other = "otherValues") %>%
  step_corr(all_numeric(), -all_outcomes(), threshold = 0.9) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes())

#data for models with filtering
train.data3 <- juice(prep(train_recipe2, retain=T))

# Neural Network ----------------------------------------------------------

set.seed(24)
nnetgrid <- expand.grid(size = seq(5, 20, 5), decay = seq(0, 0.20,0.05))
nnet.res <- as.data.frame(matrix(NA, nrow = dim(nnetgrid)[1], ncol = 4))

for(i in 1:dim(nnetgrid)[1]){
  fit.nnet <- sbf(BinaryDiagnosis ~., data = train.data3, method = "nnet", sbfControl = filterControl,
                  tuneGrid= nnetgrid[i,])
  
  nnet.res[i,] <- fit.nnet$results
  
  #keep best model
  if (i == 1){
    keep.nonlinear <- fit.nnet
  } else if (fit.nnet$results[1] > keep.nonlinear$results[1]){
    keep.nonlinear <- fit.nnet
  }
}

ggplot(varImp(keep.nonlinear$fit), mapping = NULL) + 
  ggtitle("Neural Network Variable Importance") +
  theme(text = element_text(size=18))

#write.csv(nnet.res, file="nnetres.csv")
#write.csv(nnetgrid, file="nnetgrid.csv")

# LDA ---------------------------------------------------------------------

set.seed(24)
fit.lda <- sbf(BinaryDiagnosis ~., data = train.data3, method = "lda", sbfControl = filterControl)
fit.lda$results

ggplot(varImp(fit.lda$fit), mapping = NULL) + 
  ggtitle("LDA Variable Importance") + 
  theme(text = element_text(size=18))

# Resampling plots --------------------------------------------------------
resamp <- resamples(list(GBM = keep.gbm, NeuralNet = keep.nonlinear, LDA = fit.lda))

summary(resamp)
summary(diff(resamp))
bwplot(resamp, scales=list(y=list(cex=1.5), x=list(cex=1)))
fit.lda$results

# Predictive performance on the test set ----------------------------------
final.dat <- bake(prep(train_recipe2), test.data)
sum(predict(keep.nonlinear, final.dat)[,1] == final.dat$BinaryDiagnosis)/length(final.dat$BinaryDiagnosis)
res.roc <- predict(keep.nonlinear, final.dat)
plot.roc(final.dat$BinaryDiagnosis, res.roc[,3])

# Cut off value that minimizes MCT ----------------------------------------
sens <- function(prob)  sum(res.roc[,3] >= prob & test.data$BinaryDiagnosis==1)/sum(test.data$BinaryDiagnosis==1)
spec <- function(prob)  sum(res.roc[,3] < prob & test.data$BinaryDiagnosis==0)/sum(test.data$BinaryDiagnosis==0)

MCT <- function(cut) 4 * 0.25 * (1 - sens(cut)) + 0.75 * (1-spec(cut))
optimize(MCT, interval = c(0.01, 0.99), maximum = F)
sum(as.numeric((res.roc[,3] > 0.7321)) == final.dat$BinaryDiagnosis)/length(final.dat$BinaryDiagnosis)
