detach("package:h2o", unload = TRUE)
library(h2o)

# Initiallizing H2O with 2 threads and 10 gigs of memory
h2o.shutdown()
h2o.init(nthreads = 2)



data1$BA <- (data1$BMI*data1$log_age) 
data1$BAP <- (data1$BMI*data1$log_age*data1$log_pregnant) 
data1$IPP <- (data1$serum_insulin*data1$plasma_glucose*data1$pedigree)
data1$TB <-  (data1$triceps_skin*data1$BMI)
data1$log_plasma <- log(data1$plasma_glucose+1)
data1$log_bmi <- log(data1$BMI+1)
data1$log_triceps_skin <- log(data1$triceps_skin+1)
data1$log_DBP <- log(data1$DBP+1)
data1$BMI_prob <- ifelse(data1$BMI>22.8,1,0)


testing$BA <- (testing$BMI*testing$log_age) 
testing$BAP <- (testing$BMI*testing$log_age*testing$log_pregnant) 
testing$IPP <- (testing$serum_insulin*testing$plasma_glucose*testing$pedigree)
testing$TB <-  (testing$triceps_skin*testing$BMI)
testing$log_plasma <- log(testing$plasma_glucose+1)
testing$log_bmi <- log(testing$BMI+1)
testing$log_triceps_skin <- log(testing$triceps_skin+1)
testing$log_DBP <- log(testing$DBP+1)
testing$BMI_prob <- ifelse(testing$BMI>22.8,1,0)

diabetes_h2o <- as.h2o(as.data.frame(data1))
test_h2o <- as.h2o(testing)
db.splits <- h2o.splitFrame(data =  diabetes_h2o, ratios = .8, seed = 1234)
train_glm <- db.splits[[1]]
valid_glm <- db.splits[[2]]


colnames(diabetes_h2o)
list <- c("log_plasma", "log_DBP" , "log_triceps_skin", "serum_insulin","pedigree",      
            "log_age","log_pregnant" ,"BA", "BAP", "IPP", "TB" ,"BMI_prob" )
predictors <- list
response <- "diabetes"

#--------LASSO:

db.lasso <- h2o.glm(x = predictors, 
                        y = response, 
                        training_frame = train_glm, 
                        validation_frame = valid_glm,  
                        family = 'binomial',
                        link = 'logit', 
                        alpha=1,
                        seed = 120)


h2o.auc(db.lasso,valid = TRUE)

h2o.varimp_plot(db.lasso, num_of_features = 50)


h2o.confusionMatrix(db.lasso, valid=T,thresholds=0.65)
pred_lasso1 <- as.data.frame(h2o.predict(db.lasso, test_h2o))
write.csv(pred_lasso1,"Pred4_lasso.csv")
#---------------RIDGE:

db.ridge <- h2o.glm(x = predictors, 
                        y = response, 
                        training_frame = train_glm, 
                        validation_frame = valid_glm,  
                        family = 'binomial',
                        link = 'logit', 
                        alpha=0,
                        seed = 120)


h2o.auc(db.ridge,valid = TRUE)

h2o.varimp_plot(db.ridge, num_of_features = 50)


h2o.confusionMatrix(db.ridge, valid=T,thresholds=0.65)

###############################################################################
# GRID SEARCH
################################################################################

# Hyperparameter LAMBDA and ALPHA
#lambda_seq <- seq(0.1,0.00001,by=-0.0001)
#hyper_params <- list( lambda = lambda_seq)
# hyper_params <- list( lambda = c(0.099,0.001,0.002,0.003,0.004,0.005,0.006,0.009,0.0001,0.00011))
# hyper_params <- list( lambda = c(1, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0) ,
hyper_params <- list(alpha = c(0, .25, 0.35,.5,.75,.89,.99,1))

# Grid search for selecting the best model
grid <- h2o.grid(x = predictors, y = response ,training_frame = train_glm,nfolds=5,validation_frame = valid_glm,
                 algorithm = "glm", grid_id = "db_grid", hyper_params = hyper_params,seed=120,
                 family = 'binomial',link = 'logit',  search_criteria = list(strategy = "Cartesian"))

# Sort the grid models by mse
sortedGrid <- h2o.getGrid("db_grid", sort_by = "auc", decreasing = FALSE)
sortedGrid
View(sortedGrid@summary_table)




#--------Re-tuning:

#alpha=0.5 


db.elastic1 <- h2o.glm(x = predictors, 
                       y = response, 
                       training_frame = train_glm, 
                       validation_frame = valid_glm,  
                       family = 'binomial',
                       link = 'logit', 
                       alpha=0.5,
                       lambda_search = T,
                       seed = 1120)


h2o.auc(db.elastic1,valid = TRUE)

h2o.varimp_plot(db.elastic1, num_of_features = 50)


h2o.confusionMatrix(db.elastic1, valid=T,thresholds=0.65)
pred_el1 <- as.data.frame(h2o.predict(db.elastic1, test_h2o))
submit1 <- data.frame('Id'=testing$Id,'Predicted'=pred_el1$predict,'P1'=pred_el1$p1)
write.csv(submit1,"pred5_el.csv",row.names = F)



# alpha=0.35
db.elastic2 <- h2o.glm(x = predictors, 
                           y = response, 
                           training_frame = train_glm, 
                           validation_frame = valid_glm,  
                           family = 'binomial',
                           link = 'logit', 
                           lambda_search = T,
                           alpha=0.35,
                           seed = 120)


h2o.auc(db.elastic2,valid = TRUE)

h2o.varimp_plot(credit.elastic2, num_of_features = 50)


h2o.confusionMatrix(db.elastic2, valid=T,thresholds=0.65)

pred_el3 <- as.data.frame(h2o.predict(db.elastic2, test_h2o))
submit2 <- data.frame('Id'=testing$Id,'Predicted'=pred_el3$predict,'P1'=pred_el3$p1)
write.csv(submit2,"pred11_el.csv",row.names = F)


# alpha=0.35
db.elastic3 <- h2o.glm(x = predictors, 
                       y = response, 
                       training_frame = train_glm, 
                       validation_frame = valid_glm,  
                       family = 'binomial',
                       link = 'logit', 
                       lambda_search = T,
                       alpha=0.7,
                       seed = 110)



#---------------GBM

db.gbm1 <- h2o.gbm(x = predictors, 
                   y = response, 
                   training_frame = train_glm, 
                   validation_frame = valid_glm,  
                   distribution = 'multinomial',
                   seed = 110)

h2o.auc(db.gbm1,valid = TRUE)

h2o.confusionMatrix(db.gbm1, valid=T)

pred_gbm1 <- as.data.frame(h2o.predict(db.gbm1, test_h2o))
out_gbm1 <- data.frame('Id'=testing$Id,'Predicted'=pred_gbm1$predict,'P1'=pred_gbm1$p1)
write.csv(out_gbm1,"pred11_gbm.csv",row.names = F)

#---------------GBM

db.gbm2 <- h2o.gbm(x = predictors, 
                   y = response, 
                   nfolds = 10,
                   training_frame = train_glm, 
                   validation_frame = valid_glm,  
                   distribution = 'multinomial',
                   seed = 190)

h2o.auc(db.gbm2,valid = TRUE)

h2o.confusionMatrix(db.gbm2, valid=T,thresholds=0.65)

pred_gbm1 <- as.data.frame(h2o.predict(db.gbm1, test_h2o))
out_gbm1 <- data.frame('Id'=testing$Id,'Predicted'=pred_gbm1$predict,'P1'=pred_gbm1$p1)
write.csv(out_gbm1,"pred11_gbm.csv",row.names = F)


#---------------------------------------------------- FINAL MODEL

# Shutting down the H2o cluster
h2o.shutdown()

