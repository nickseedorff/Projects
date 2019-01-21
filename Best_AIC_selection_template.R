##################################################################
#Nick Seedorff
#October 16th
#Best subset AIC selection with Linear Mixed Model 
##################################################################

#Packages
library(lme4)
library(multcomp)

#Call data
data("mtcars")
head(mtcars)

#Define a vector with the variables you want to use (including the outcome)
names(mtcars)
use.var  <- c("mpg", "cyl", "disp", "qsec", "carb", "gear")

#Data for modeling
dat.use <- mtcars[,use.var]

#Creates a matrix of indicators, uses this to select which variables to include in the models
subset <- expand.grid(2, c(0,3), c(0,4), c(0,5))
#The "2" means cyl will be included in every model
#Excluded 1 because it is the outcome (Used in line 37)
#Excluded 6 because gear is a a random effect (Used in line 37)

#Create a vector to store AIC results
aic.res <- matrix(NA, nrow= dim(subset)[1], ncol=1)

#Loop over all possible models (dimension is number of rows of the subset matrix)
for(i in 1:dim(subset)[1]){
  #string with variables to include in the model, separated by +
  var.use <- paste(colnames(dat.use)[unlist(subset[i,])], collapse = "+")
  
  #Convert string to a formula so that it can be used when calling the model
  form <- as.formula(paste("mpg ~ (1 |gear) + ", var.use))
  
  #Call model, store AIC, use ML during selection phase
  aic.res[i,1] <- AIC(lmer(form, data=dat.use, REML=F))
}

#Best 6 AICs
head(aic.res[order(aic.res)])
#Row numer of 6 best models
head(order(aic.res))

# Final model -------------------------------------------------------------

#8 was the best model based on lowest AIC
i=8
var.final <- paste( colnames(dat.use)[unlist(subset[i,])],  collapse = "+")
form.final <- as.formula(paste("mpg ~ (1 |gear) +", var.final))

#fit final model using REML
mod.final <- lmer(form.final, data=dat.use, REML=T)
summary(mod.final)

#Test significance by assuming asympoptic normality of the beta estimates
cftest(mod.final)
