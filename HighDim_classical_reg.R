#############################################################################
#Nick Seedorff
#1/14/2019
#Description:
#Simulation study to look at classical techniques in high dimensional setting
#Simulation implements p-value based model selection and looks at performance
#metrics of the final models as the number of covariates increases.
#############################################################################

#Packages
library(ggplot2)

#Marginal regression function for model selection
lm_func <- function(feature){
  lm_mod <- summary(lm(y ~ feature)) #Regression model
  return(lm_mod$coefficients[2,4]) #Return p-value
}

#Define variables
N <- 1000
n <- 50
p <- c(5, 50, 500, 5000)

#Matrix for results
results <- matrix(NA, ncol = length(p) * 3, nrow = N)

#Set progress bar, runtime ~ 1.5 hours
pb <- txtProgressBar(min = 0, max = N, style = 3)
#Loop through number of covariates
for(j in 1:length(p)){
  #N simulations for each number of covariates
  for(i in 1:N){
    #Generate the data
    x <- matrix(runif(n * p[j] , 0 ,1), ncol = p[j]) #Design matrix, iid so order doesn't matter
    y <- rnorm(n, 0 , 1) #Outcome
    
    #Generate data to calculate MSPE with
    xx <- matrix(runif(n * p[j] , 0 ,1), ncol = p[j]) #Design matrix, iid so order doesn't matter
    yy <- rnorm(n, 0 , 1) #Outcome
    
    #Model selection
    p_vals <- apply(x, 2, lm_func) #p-values
    x2 <- as.data.frame(x[, order(p_vals)[1:5]]) #df of chosen features
    xx2 <- as.data.frame(xx[, order(p_vals)[1:5]]) #df of chosen features for MSPE calculation
    
    #Run final model, store results
    lm_final <- lm(y ~ ., data=as.data.frame(x2)) #Final models
    results[i, (j-1) * 3 + 1] <- mean(summary(lm_final)$coef[-1, 1]^2) #MSE
    results[i, (j-1) * 3 + 2] <- crossprod(yy - predict(lm_final, xx2))/n #MSPE
    results[i, (j-1) * 3 + 3] <- mean((summary(lm_final)$coef[-1, 4] >= 0.05) * 1) #Coverage
    
    #update progress bar
    setTxtProgressBar(pb, i)
  }
}

#Store results data for future use
#write.csv(results, "C:\\Users\\Nick\\Dropbox\\High Dimensional\\hw1_data.csv")
#results <- read.csv("C:\\Users\\Nick\\Dropbox\\High Dimensional\\hw1_data.csv")

# Bar plots of means ------------------------------------------------------

#Means for each variable by group
col_means <- data.frame(colMeans(results, na.rm = T), as.factor(rep(c("MSE", "MSPE", "Coverage"), times = length(p))),
                                                        as.factor(rep(c("5", "50", "500", "5000"), each = 3)))
colnames(col_means) <- c("Mean", "Metric", "Num_Covariates")

#Bar plot by group
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
ggplot(col_means, aes(x = Metric , y = Mean)) +   
  geom_bar(aes(fill = Num_Covariates), position = "dodge", stat="identity") +
  scale_fill_manual(values=cbPalette) + 
  theme(axis.text.x = element_text(size = 18), axis.text.y = element_text(size = 18),
        axis.title.y = element_text(size = 18), axis.title.x = element_blank(), legend.text=element_text(size=16)) 
