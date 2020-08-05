
library(dplyr)

#setwd("P:/Courses/MMA 869 - Machine Learning and  AI/Session 6 - Kaggle/Kaggle/data")

train <- read.csv("diabetes_train.csv")
test <- read.csv("diabetes_test.csv")

data <- rbind(train,test)
str(data)

var_Summ=function(x){
  
  if(class(x)=="numeric"){
    Var_Type=class(x)
    n<-length(x)
    nmiss<-sum(is.na(x))
    mean<-mean(x,na.rm=T)
    std<-sd(x,na.rm=T)
    var<-var(x,na.rm=T)
    min<-min(x,na.rm=T)
    max<-max(x,na.rm=T)
    return(c(Var_Type=Var_Type, n=n,nmiss=nmiss,mean=mean,std=std,var=var,min=min,max=max))
  }
  else{
    Var_Type=class(x)
    n<-length(x)
    nmiss<-sum(is.na(x))
    
    return(c(Var_Type=Var_Type, n=n,nmiss=nmiss))
  }
  
}

#Applying above defined function on numerical variables

num_var= sapply(data,is.numeric)
Other_var= !sapply(data,is.numeric)

my_num_vars<-t(data.frame(apply(data[num_var], 2, var_Summ)))
my_cat_vars<-t(data.frame(apply(data[Other_var], 2, var_Summ)))

#Apply to test

#Create log of age
hist(train$age)
data$log_age <- log(data$age+1)
hist(data$log_age)


#Num pregnant normalize
hist(data$num_times_pregnant)
data$log_pregnant <- log(data$num_times_pregnant+1)
hist(data$log_pregnant)

#Impute BMI 0 with average
hist(data$BMI)
data$BMI[data$BMI==0] <- mean(data$BMI) 
hist(data$BMI)


#Impute plasma_glucose
hist(data$plasma_glucose)
data$plasma_glucose[data$plasma_glucose==0] <- mean(data$plasma_glucose) 
hist(data$plasma_glucose)

#Impute DBP
hist(data$DBP)
data$DBP[data$DBP==0] <- median(data$DBP) 
hist(data$DBP)

#Impute Insulin Serum
hist(data$serum_insulin)
data$serum_insulin[data$serum_insulin==0] <- mean(data$serum_insulin)
hist(data$serum_insulin)
#Impute triceps skin
hist(data$triceps_skin)
data$triceps_skin[data$triceps_skin==0] <- median(data$triceps_skin) 
hist(data$triceps_skin)

#num_var2= sapply(test,is.numeric)
#Other_var2= !sapply(test,is.numeric)

#my_num_vars2<-t(data.frame(apply(test[num_var2], 2, var_Summ)))
#my_cat_vars2<-t(data.frame(apply(test[Other_var2], 2, var_Summ)))

if("pacman" %in% rownames(installed.packages()) == FALSE) {install.packages("pacman")} # Check if you have universal installer package, install if not

pacman::p_load("caret","ROCR","lift","glmnet","MASS","e1071") #Check, and if needed install the necessary packages

#Split data into test & train
testing <- data[577:768,]
data1 <- data[1:576,]

#Convert response to factor
data1$diabetes <- as.factor(data1$diabetes)
data1 <- data1 %>% dplyr::select(-c(num_times_pregnant,age,Id))
set.seed(110) #set a random number generation seed to ensure that the split is the same everytime
inTrain <- createDataPartition(y = data1$diabetes,
                               p = 0.75, list = FALSE)
training <- data1[ inTrain,]
valid <- data1[ -inTrain,]

#-------------Logistic Model
testing$diabetes <- NULL

model_logistic<-glm(diabetes~ plasma_glucose+log(DBP+1)+triceps_skin+serum_insulin+BMI+log(1+pedigree)
                    +log_age+log_pregnant , data=training, family="binomial"(link="logit"))
summary(model_logistic)

model_logistic_stepwiseAIC<-stepAIC(model_logistic,direction = c("both"),trace = 1) #AIC stepwise
summary(model_logistic_stepwiseAIC) 

par(mfrow=c(1,4))
plot(model_logistic_stepwiseAIC) #Error plots: similar nature to lm plots
par(mfrow=c(1,1))

###Finding predicitons: probabilities and classification
logistic_probabilities<-predict(model_logistic_stepwiseAIC,newdata=valid,type="response") #Predict probabilities
logistic_classification<-rep("1",143)
logistic_classification[logistic_probabilities<0.65]="0" #Predict classification using 0.6073 threshold. Why 0.6073 - that's the average probability of being retained in the data. An alternative code: logistic_classification <- as.integer(logistic_probabilities > mean(testing$Retained.in.2012. == "1"))
logistic_classification<-as.factor(logistic_classification)
###Confusion matrix  
confusionMatrix(logistic_classification,valid$diabetes,positive = "1") #Display confusion matrix

####ROC Curve
logistic_ROC_prediction <- prediction(logistic_probabilities, valid$diabetes)
logistic_ROC <- performance(logistic_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(logistic_ROC) #Plot ROC curve

####AUC (area under curve)
auc.tmp <- performance(logistic_ROC_prediction,"auc") #Create AUC data
logistic_auc_valid <- as.numeric(auc.tmp@y.values) #Calculate AUC
logistic_auc_valid #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value

#### Lift chart
plotLift(logistic_probabilities, testing$Retained.in.2012., cumulative = TRUE, n.buckets = 10) # Plot Lift chart


#Prediction on test data
logistic_probabilities<-predict(model_logistic_stepwiseAIC,newdata=testing,type="response") #Predict probabilities
logistic_classification<-rep("1",192)
logistic_classification[logistic_probabilities<0.65]="0" #Predict classification using 0.6073 threshold. Why 0.6073 - that's the average probability of being retained in the data. An alternative code: logistic_classification <- as.integer(logistic_probabilities > mean(testing$Retained.in.2012. == "1"))
logistic_classification<-as.factor(logistic_classification)
pred1 <- data.frame('Id'=testing$Id, 'Predicted'=logistic_classification)
write.csv(pred1, "Pred1_logistic.csv", row.names = F)

#---------LOGISTIC 2

#-------------Logistic Model
testing$diabetes <- NULL

model_logistic2<-glm(diabetes~ plasma_glucose+log(DBP+1)+triceps_skin+serum_insulin+BMI+log(1+pedigree)
                    +log_age+log_pregnant + (BMI*log_age) + (BMI*log_age*log_pregnant) , data=training, family="binomial"(link="logit"))
summary(model_logistic2)

model_logistic_stepwiseAIC2<-stepAIC(model_logistic2,direction = c("both"),trace = 1) #AIC stepwise
summary(model_logistic_stepwiseAIC2) 


###Finding predicitons: probabilities and classification
logistic_probabilities2<-predict(model_logistic_stepwiseAIC2,newdata=valid,type="response") #Predict probabilities
logistic_classification2<-rep("1",143)
logistic_classification2[logistic_probabilities2<0.65]="0" #Predict classification using 0.6073 threshold. Why 0.6073 - that's the average probability of being retained in the data. An alternative code: logistic_classification <- as.integer(logistic_probabilities > mean(testing$Retained.in.2012. == "1"))
logistic_classification2<-as.factor(logistic_classification2)
###Confusion matrix  
confusionMatrix(logistic_classification2,valid$diabetes,positive = "1") #Display confusion matrix

####ROC Curve
logistic_ROC_prediction2 <- prediction(logistic_probabilities2, valid$diabetes)
logistic_ROC2 <- performance(logistic_ROC_prediction2,"tpr","fpr") #Create ROC curve data
plot(logistic_ROC2) #Plot ROC curve

####AUC (area under curve)
auc.tmp2 <- performance(logistic_ROC_prediction2,"auc") #Create AUC data
logistic_auc_valid2 <- as.numeric(auc.tmp2@y.values) #Calculate AUC
logistic_auc_valid2 #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value


#Prediction on test data
logistic_probabilities_2<-predict(model_logistic_stepwiseAIC2,newdata=testing,type="response") #Predict probabilities
logistic_classification_2<-rep("1",192)
logistic_classification_2[logistic_probabilities_2<0.65]="0" #Predict classification using 0.6073 threshold. Why 0.6073 - that's the average probability of being retained in the data. An alternative code: logistic_classification <- as.integer(logistic_probabilities > mean(testing$Retained.in.2012. == "1"))
logistic_classification_2<-as.factor(logistic_classification_2)
pred2 <- data.frame('Id'=testing$Id, 'Predicted'=logistic_classification_2)
write.csv(pred2, "Pred2_logistic.csv", row.names = F)



#---------LOGISTIC 3

model_logistic3<-glm(diabetes~ plasma_glucose+log(DBP+1)+triceps_skin+serum_insulin+BMI+log(1+pedigree)
                     +log_age+log_pregnant + (BMI*log_age) + (BMI*log_age*log_pregnant) +
                       (serum_insulin*plasma_glucose*pedigree)+
                       (triceps_skin*BMI), data=training, family="binomial"(link="logit"))
summary(model_logistic3)

model_logistic_stepwiseAIC3<-stepAIC(model_logistic3,direction = c("both"),trace = 1) #AIC stepwise
summary(model_logistic_stepwiseAIC3) 


###Finding predicitons: probabilities and classification
logistic_probabilities3<-predict(model_logistic_stepwiseAIC3,newdata=valid,type="response") #Predict probabilities
logistic_classification3<-rep("1",143)
logistic_classification3[logistic_probabilities3<0.65]="0" #Predict classification using 0.6073 threshold. Why 0.6073 - that's the average probability of being retained in the data. An alternative code: logistic_classification <- as.integer(logistic_probabilities > mean(testing$Retained.in.2012. == "1"))
logistic_classification3<-as.factor(logistic_classification3)
###Confusion matrix  
confusionMatrix(logistic_classification3,valid$diabetes,positive = "1") #Display confusion matrix

####ROC Curve
logistic_ROC_prediction3 <- prediction(logistic_probabilities3, valid$diabetes)
logistic_ROC3 <- performance(logistic_ROC_prediction3,"tpr","fpr") #Create ROC curve data
plot(logistic_ROC3) #Plot ROC curve

####AUC (area under curve)
auc.tmp3 <- performance(logistic_ROC_prediction3,"auc") #Create AUC data
logistic_auc_valid3 <- as.numeric(auc.tmp3@y.values) #Calculate AUC
logistic_auc_valid3 #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value


#Prediction on test data
logistic_probabilities_3<-predict(model_logistic_stepwiseAIC3,newdata=testing,type="response") #Predict probabilities
logistic_classification_3<-rep("1",192)
logistic_classification_3[logistic_probabilities_3<0.65]="0" #Predict classification using 0.6073 threshold. Why 0.6073 - that's the average probability of being retained in the data. An alternative code: logistic_classification <- as.integer(logistic_probabilities > mean(testing$Retained.in.2012. == "1"))
logistic_classification_3<-as.factor(logistic_classification_3)
pred3 <- data.frame('Id'=testing$Id, 'Predicted'=logistic_classification_3)
write.csv(pred3, "Pred3_logistic.csv", row.names = F)

#---------LOGISTIC 4

model_logistic4<-glm(diabetes~ log_bmi+log(1+pedigree)+log_pregnant+log_age+BA+BAP+IPP+log_plasma+
                       log_triceps_skin+log_DBP+BMI_prob
                     , data=training, family="binomial"(link="logit"))
summary(model_logistic4)

model_logistic_stepwiseAIC4<-stepAIC(model_logistic4,direction = c("both"),trace = 1) #AIC stepwise
summary(model_logistic_stepwiseAIC4) 


###Finding predicitons: probabilities and classification
logistic_probabilities4<-predict(model_logistic_stepwiseAIC4,newdata=valid,type="response") #Predict probabilities
logistic_classification4<-rep("1",143)
logistic_classification4[logistic_probabilities4<0.65]="0" #Predict classification using 0.6073 threshold. Why 0.6073 - that's the average probability of being retained in the data. An alternative code: logistic_classification <- as.integer(logistic_probabilities > mean(testing$Retained.in.2012. == "1"))
logistic_classification4<-as.factor(logistic_classification4)
###Confusion matrix  
confusionMatrix(logistic_classification4,valid$diabetes,positive = "1") #Display confusion matrix

####ROC Curve
logistic_ROC_prediction4 <- prediction(logistic_probabilities4, valid$diabetes)
logistic_ROC4 <- performance(logistic_ROC_prediction4,"tpr","fpr") #Create ROC curve data
plot(logistic_ROC4) #Plot ROC curve

####AUC (area under curve)
auc.tmp4 <- performance(logistic_ROC_prediction4,"auc") #Create AUC data
logistic_auc_valid4 <- as.numeric(auc.tmp4@y.values) #Calculate AUC
logistic_auc_valid4 #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value


#Prediction on test data
logistic_probabilities_4<-predict(model_logistic_stepwiseAIC4,newdata=testing,type="response") #Predict probabilities
logistic_classification_4<-rep("1",192)
logistic_classification_4[logistic_probabilities_4<0.65]="0" #Predict classification using 0.6073 threshold. Why 0.6073 - that's the average probability of being retained in the data. An alternative code: logistic_classification <- as.integer(logistic_probabilities > mean(testing$Retained.in.2012. == "1"))
logistic_classification_4<-as.factor(logistic_classification_4)
pred4 <- data.frame('Id'=testing$Id, 'Predicted'=logistic_classification_4)
write.csv(pred4, "Pred6_logistic.csv", row.names = F)

#------------XGBOOST

if("pacman" %in% rownames(installed.packages()) == FALSE) {install.packages("pacman")} # Check if you have universal installer package, install if not

pacman::p_load("caret","ROCR","lift","xgboost") #Check, and if needed install the necessary packages


db_matrix <- model.matrix(diabetes~ ., data = data1)[,-1]

x_train <- db_matrix[ inTrain,]
x_test <- db_matrix[ -inTrain,]

y_train <-training$diabetes
y_test <-valid$diabetes

model_XGboost<-xgboost(data = data.matrix(x_train), 
                       label = as.numeric(as.character(y_train)), 
                       eta = 0.1,       # hyperparameter: learning rate 
                       max_depth = 15,  # hyperparameter: size of a tree in each boosting iteration
                       nround=25,       # hyperparameter: number of boosting iterations  
                       objective = "binary:logistic"
)

XGboost_prediction<-predict(model_XGboost,newdata=x_test, type="response") #Predict classification (for confusion matrix)
confusionMatrix(as.factor(ifelse(XGboost_prediction>0.65,1,0)),y_test,positive="1") #Display confusion matrix

####ROC Curve
XGboost_ROC_prediction <- prediction(XGboost_prediction, y_test) #Calculate errors
XGboost_ROC_testing <- performance(XGboost_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(XGboost_ROC_testing) #Plot ROC curve

####AUC
auc.tmp <- performance(XGboost_ROC_prediction,"auc") #Create AUC data
XGboost_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
XGboost_auc_testing #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value

#### Lift chart
plotLift(XGboost_prediction, y_test, cumulative = TRUE, n.buckets = 10) # Plot Lift chart

### An alternative way is to plot a Lift curve not by buckets, but on all data points
Lift_XGboost <- performance(XGboost_ROC_prediction,"lift","rpp")
plot(Lift_XGboost)


pred4 <- data.frame('Id'=testing$Id, 'Predicted'=logistic_classification_4)
write.csv(pred4, "Pred6_logistic.csv", row.names = F)
