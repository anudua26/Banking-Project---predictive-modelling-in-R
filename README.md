[bank-full_train (1).csv](https://github.com/anudua26/Banking-Project---predictive-modelling-in-R/files/8294800/bank-full_train.1.csv)
[bank-full_train.csv](https://github.com/anudua26/Banking-Project---predictive-modelling-in-R/files/8294840/bank-full_train.csv)
[Prediction using model.csv](https://github.com/anudua26/Banking-Project---predictive-modelling-in-R/files/8294867/Prediction.using.model.csv)
# Banking-Project---predictive-modelling-in-R

Business Scenario:

A local bank is rolling out term deposit for its customers. They have in the past connected to their customer base through phone calls. Results for these previous campaigns were recorded and have been provided to the current campaign manager to use the same in making this campaign more effective.

Challenges that the manager faces are following:

•  Customers have recently started to complain that bank’s marketing staff bothers them with irrelevant product calls and this should immediately stop

•  There is no prior framework for her decide and choose which customer to call and which one to leave alone

She has decided to use past data to automate this decision, instead of manually choosing through each and every customer. Previous campaign data which has been made available to her; contains customer characteristics , campaign characteristics, previous campaign information as well as whether customer ended up subscribing to the product as a result of that campaign or not. Using this she plans to develop a statistical model which given this information predicts whether customer in question will subscribe to the product or not. A successful model which is able to do this, will make her campaign efficiently targeted and less bothering to uninterested customers.

We are given you two datasets - bank-full_train.csv and bank-full_test.csv . Need to use data bank-full_train to build predictive model for response variable “y”. bank-full_test data contains all other factors except “y”, you need to predict that using the model that you developed and submit your predicted values in a csv files.

Evaluation Criterion :KS score on test data. larger KS, better Model

The predictions should not contain any NA values. Can use any predictive modelling technique


DESIRED RESULT from this project:  

KS score for test data should come out to be more than 0.47


My R script for this project as below:

setwd("C:/Users/Anupam/Desktop/Data")
getwd()
bd_train = read.csv("bank-full_train.csv", stringsAsFactors = F)
bd_train
bd_test=read.csv("bank-full_test.csv", stringsAsFactors = F)
bd_test
View(bd_train)
View(bd_test)
library(dplyr)
glimpse(bd_train)
bd_test$y = NA
View(bd_test)
bd_train$data ="train"
bd_test$data = "test"

bd = rbind(bd_train, bd_test)
glimpse(bd)


CreateDummies=function(data,var,freq_cutoff=0){
  t=table(data[,var])
  t=t[t>freq_cutoff]
  t=sort(t)
  categories=names(t)[-1]
  for( cat in categories){
    name=paste(var,cat,sep="_")
    name=gsub(" ","",name)
    name=gsub("-","_",name)
    name=gsub("\\?","Q",name)
    name=gsub("<","LT_",name)
    name=gsub("\\+","",name)
    name=gsub("\\/","_",name)
    name=gsub(">","GT_",name)
    name=gsub("=","EQ_",name)
    name=gsub(",","",name)
    data[,name]=as.numeric(data[,var]==cat)
  }
  data[,var]=NULL
  return(data)
}

sapply(bd, function(x) is.character (x))

names(bd)[sapply(bd,function(x) is.character(x))]

cat_cols = c("job","marital","education","default", "housing", "loan", "contact", 
             "month", "poutcome", "y")

for(cat in cat_cols){
  bd = CreateDummies(bd,cat,100)
  
}
glimpse(bd)


#Missing values

sum(is.na(bd))

for(col in names(bd)){
  
  if(sum(is.na (bd[,col]))>0 & !(col %in% c("data","y_no"))){
    
    bd[is.na(bd[,col]),col]=mean(bd[,col],na.rm=T)
  }
  
}
sum(is.na(bd))


bd_train = bd[bd$data == "train", ]

bd_test = bd[bd$data == "test", ]
bd_train$data = NULL
bd_test$data = NULL



#Building the model
set.seed(2)
s = sample(1:nrow(bd_train), 0.8*nrow(bd_train))

bd_train1 = bd_train[s,]

bd_train2 = bd_train[-s,]

glimpse(bd)
 
# Multicollinearnity for bd_train1
 
 library(car)
 
 for_vif=lm(y_no~.-ID,data=bd_train1)
 summary(for_vif)
 sort(vif(for_vif), decreasing = TRUE) [1:3]
 
 for_vif=lm(y_no~.-ID-month_may,data=bd_train1)
 sort(vif(for_vif), decreasing = TRUE) [1:3]

 for_vif=lm(y_no~.-ID-month_may-job_blue_collar,data=bd_train1)
 sort(vif(for_vif), decreasing = TRUE) [1:3]
 
 
 #Logistic regression
 
 formula(for_vif)

log_fit = glm(y_no~.-ID-month_may-job_blue_collar,data = bd_train1, family = "binomial")
summary(log_fit)
log_fit = step(log_fit)
formula(log_fit)

log_fit = glm(y_no ~ balance + day + duration + campaign + job_student + job_housemaid + 
                job_retired + job_admin. + job_technician + job_management + 
                marital_married + education_primary + housing_yes + loan_no + 
                contact_unknown + month_mar + month_sep + month_oct + month_jan + 
                month_feb + month_apr + month_nov + month_jun + month_aug + 
                month_jul + poutcome_other + poutcome_failure + poutcome_unknown,data = bd_train1, family = "binomial")
summary(log_fit)

library(pROC)

val.score=predict(log_fit,newdata = bd_train2,type="response")
auc_score=auc(roc(bd_train2$y_no,val.score))
auc_score


# Now we build the model on the entire training data

for_vif.final=lm(y_no~.-ID-month_may-job_blue_collar,data=bd_train)
summary(for_vif.final)
sort(vif(for_vif.final), decreasing = TRUE) [1:3]


log_fit.final = glm(y_no~.-ID-month_may-job_blue_collar,data=bd_train, family = "binomial")
summary(log_fit.final)
log_fit.final = step(log_fit.final)

formula(log_fit.final)
log_fit.final = glm(y_no ~ balance + day + duration + campaign + job_student + job_housemaid + 
                      job_retired + job_admin. + job_technician + job_management + 
                      marital_married + education_primary + education_tertiary + 
                      housing_yes + loan_no + contact_unknown + 
                      month_mar + month_sep + month_oct + month_jan + month_feb + 
                      month_apr + month_nov + month_jun + month_aug + month_jul + 
                      poutcome_other + poutcome_failure + poutcome_unknown, data = bd_train, family = "binomial")

summary(log_fit.final)

train.score=predict(log_fit.final,newdata = bd_train,type="response")
real=bd_train$y_no
cutoff = 0.06
predicted=as.numeric(train.score>cutoff)
cutoffs=seq(0.001,0.999,0.001)
cutoff_data=data.frame(cutoff=99,Sn=99,Sp=99,KS=99,F5=99,F.1=99,M=99)
for(cutoff in cutoffs){
  predicted=as.numeric(train.score>cutoff)
  TP=sum(real==1 & predicted==1)
  TN=sum(real==0 & predicted==0)
  FP=sum(real==0 & predicted==1)
  FN=sum(real==1 & predicted==0)
  P=TP+FN
  N=TN+FP
  Sn=TP/P
  Sp=TN/N
  precision=TP/(TP+FP)
  recall=Sn
  KS=(TP/P)-(FP/N)
  
  F5=(26*precision*recall)/((25*precision)+recall)
  F.1=(1.01*precision*recall)/((.01*precision)+recall)
  
  cutoff_data=rbind(cutoff_data,c(cutoff,Sn,Sp,KS,F5,F.1))
}
cutoff_data=cutoff_data[-1,]

which.max(cutoff_data$KS)

cutoff = cutoff_data$cutoff[which.max(cutoff_data$KS)]


cutoff_data[which.max(cutoff_data$KS),]

test.score = predict(log_fit.final,newdata=bd_test,type="response")

test.class=as.numeric(test.score>cutoff)
test.class=ifelse(test.class==1,"Yes","No")

# For project submission

write.csv(test.class,"Anupam_Dua_P5_part2.csv",row.names = F)




