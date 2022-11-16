# Importing Required Libraries --------------------------------------------
library(faraway)#for checking multicolinearity 
library(caret)  #for cross validation
library(ggcorrplot)#for data visualisation 
library(e1071)#Required for SVM algo
library(party)#Required for DT algo
library(randomForest)#Required for Random Forest Algo
library(quantmod)
library(gbm)#Required for Gredient boost
library(DMwR)# for SMOTE sampling
library(naivebayes)

# Importing data set ------------------------------------------------------
data<-read.csv("diabetes.csv",header = TRUE)
summary(data)
View(data)
dim(data)


# Checking VIF scores for Co-linearity existence  --------------------------
model<-lm(data$Outcome~.,data = data)
vif_score<-vif(model)
vif_score 

#Statistcal Analysis--------------------------------------------------------
av<-aov(model)#anova test
summary(av)
x<-cor(data) #Correlation Matrix
x

# Data Visualization  -----------------------------------------------------

ggcorrplot(cor(data))  #Generating Heat Map
hist(data$Pregnancies,data = data,labels=TRUE,xlab = "Pregnancies")
boxplot(data$Age,las=1,xlab = "Age ")
boxplot(data$Age~data$Pregnancies,las=1,xlab = "Pregnancies ",ylab = "Age")
plot(data$Insulin,data$SkinThickness,pch=19,col="lightblue",xlab="Insulin",ylab = "Skin Thickness")
abline(lm(data$SkinThickness~data$Insulin), col = "red", lwd = 3)
text(paste("Correlation:", round(cor(data$Insulin,data$SkinThickness), 2)), x = 150, y = 95)


# Checking Data Imbalance -------------------------------------------------
x<-with(data,{print(table(data$Outcome))})
x<-as.data.frame(x)
colnames(x) <- c('Class','Freq')
x
ggplot(data=x, aes(x=Class, y=Freq),xlab="Class",ylab="Frequency",title="Class Imbalance") +
  geom_bar(stat="identity", fill="steelblue")+
  geom_text(aes(label=Freq), vjust=1.6, color="white", size=3.5)+
  theme_minimal() 

?svm()

# Applying Supervised Algorithms ------------------------------------------
data$Outcome<-as.factor(data$Outcome)
train.index <- createDataPartition(data[,"Outcome"],p=0.8,list=FALSE)
data.trn <- data[train.index,]
data.tst <- data[-train.index,]
ctrl  <- trainControl(method  = "cv",number  = 10) #, summaryFunction = multiClassSummary

#KNN------------------------------------------------------------------------
fit.cv <- train(Outcome ~ ., data = data.trn, method = "knn",
  trControl = ctrl, 
  preProcess = c("center","scale"), 
  tuneGrid =data.frame(k=10))

pred_knn <- predict(fit.cv,data.tst)
confusionMatrix(table(data.tst[,"Outcome"],pred_knn))
print(fit.cv)


#Logistic Regression---------------------------------------------------------
fit.cv <- train(Outcome ~ ., data = data.trn, method = "glm",
                trControl = ctrl,tuneLength=10)

pred_lr <- predict(fit.cv,data.tst)
confusionMatrix(table(data.tst[,"Outcome"],pred_lr))
print(fit.cv)

#SVM--------------------------------------------------------------------------
svm_model<- svm(data.trn$Outcome~.,data = data.trn,cost=100,gamma=0.2)
summary(svm_model)
#confusion matrix and mis-classification error of model for train data
pred_svm_tr<-predict(svm_model,data.trn)
confusionMatrix(table(data.trn[,"Outcome"],pred_svm_tr))
#confusion matrix and mis-classification error of model for test data
pred_svm_ts<-predict(svm_model,data.tst)
confusionMatrix(table(data.tst[,"Outcome"],pred_svm_ts))

#DT---------------------------------------------------------------------------
tree<-ctree(data.trn$Outcome~.,data = data.trn,controls = ctree_control(mincriterion = 0.99,minsplit =300 ))
plot(tree)
#confusion matrix and mis-classification error of model for test data
pred_dt_tr<-predict(tree,data.trn)
confusionMatrix(table(data.trn[,"Outcome"],pred_dt_tr))
#confusion matrix and mis-classification error of model for test data
pred_dt_ts<-predict(tree,data.tst)
confusionMatrix(table(data.tst[,"Outcome"],pred_dt_ts))

#Random Forest----------------------------------------------------------------
set.seed(222)
rf<-randomForest(data.trn$Outcome~.,data = data.trn)
print(rf)
#confusion matrix and mis-classification error of model for train data
pred_rf_tr<-predict(rf,data.trn)
confusionMatrix(table(data.trn[,"Outcome"],pred_rf_tr))
#confusion matrix and mis-classification error of model for test data
pred_rf_ts<-predict(rf,data.tst)
confusionMatrix(table(data.tst[,"Outcome"],pred_rf_ts))

#Naive Bayes Classifier------------------------------------------------------
nb<-naive_bayes(data.trn$Outcome~.,data=data.trn)
#confusion matrix and mis-classification error of model for train data
pred_nb_tr<-predict(nb,data.trn)
confusionMatrix(table(data.trn[,"Outcome"],pred_nb_tr))
#confusion matrix and mis-classification error of model for test data
pred_nb_ts<-predict(nb,data.tst)
confusionMatrix(table(data.tst[,"Outcome"],pred_nb_ts))
plot(nb)



# SMOTE sampling to deal with class imbalance---------------------------
table(data.trn$Outcome)
data.trn$Outcome<-as.factor(data.trn$Outcome)
final_train<-SMOTE(data.trn$Outcome~.,data.trn,perc.over = 100,perc.under = 200)
table(final_train$Outcome)
View(final_train)

table(data.tst$Outcome)
data.tst$Outcome<-as.factor(data.tst$Outcome)
final_test<-SMOTE(data.tst$Outcome~.,data.tst,perc.over = 100,perc.under = 200)
table(final_test$Outcome)
View(final_test)


# Applying Supervised Algorithm After SMOTE -------------------------------

#KNN After SMOTE
ctrl  <- trainControl(method  = "cv",number  = 10) 
fit.cv <- train(Outcome ~ ., data = final_train, method = "knn",
  trControl = ctrl, 
  preProcess = c("center","scale"), 
  tuneGrid =data.frame(k=10))
pred_Kn <- predict(fit.cv,final_test)
confusionMatrix(table(final_test[,"Outcome"],pred_Kn))
print(fit.cv)


#Logistic Regression After SMOTE
fit.cv <- train(Outcome ~ ., data = final_train, method = "glm",
                trControl = ctrl,tuneLength=10)

pred_lr <- predict(fit.cv,final_test)
confusionMatrix(table(final_test[,"Outcome"],pred_lr))
print(fit.cv)

#SVM after SMOTE-----------------------------------------------------------
svm_model<- svm(final_train$Outcome~.,data = final_train,cost=100,gamma=0.2)
summary(svm_model)
#confusion matrix and mis-classification error of model for train data
pred_svm_tr<-predict(svm_model,final_train)
confusionMatrix(table(final_train[,"Outcome"],pred_svm_tr))
#confusion matrix and mis-classification error of model for test data
pred_svm_ts<-predict(svm_model,final_test)
confusionMatrix(table(final_test[,"Outcome"],pred_svm_ts))

#DT after SMOTE-------------------------------------------------------------
tree<-ctree(final_train$Outcome~.,data = final_train,controls = ctree_control(mincriterion = 0.99,minsplit =300 ))
plot(tree)
#confusion matrix and mis-classification error of model for test data
pred_dt_tr<-predict(tree,final_train)
confusionMatrix(table(final_train[,"Outcome"],pred_dt_tr))
#confusion matrix and mis-classification error of model for test data
pred_dt_ts<-predict(tree,final_test)
confusionMatrix(table(final_test[,"Outcome"],pred_dt_ts))

#Random Forest after SMOTE---------------------------------------------------
set.seed(222)
rf<-randomForest(final_train$Outcome~.,data = final_train)
print(rf)
#confusion matrix and mis-classification error of model for train data
pred_rf_tr<-predict(rf,final_train)
confusionMatrix(table(final_train[,"Outcome"],pred_rf_tr))
#confusion matrix and mis-classification error of model for test data
pred_rf_ts<-predict(rf,final_test)
confusionMatrix(table(final_test[,"Outcome"],pred_rf_ts))





# Problem 2 : Patient Segmentation ---------------------------------------
#hierarchical clustering ---------
x<-data[sapply(data, is.numeric)]
View(x)
y<-x

#Normalisation
m<-apply(x,2,mean)
s<-apply(x,2,sd)
x<-scale(x,m,s)

#Calculating Eucledian Distance
distance<-dist(x)
print(distance,digits=3)

#Cluster Dendogram for complete Linkage
hc.c<-hclust(distance)

hc.a<-hclust(distance,method = "complete")
plot(hc.c)

#Scree Plot to decide Number of clusters
wss<-(nrow(x)-1)*sum(apply(x[,], 2, var))
for(i in 2:20)wss[i]<-sum(kmeans(x[,],centers = i)$withinss)
plot(1:20,wss,type = "b",xlab = "Number of Clusters",ylab = "Within Group SS")

#cluster Members
member.c<-cutree(hc.c,4)
table(member.c)

#cluster Mean
print(aggregate(x,list(member.c),mean),digits = 2)
aggregate(y,list(member.c),mean)

#Plotting clusters
fviz_cluster(list(data=x,cluster=member.c))






# K-means -----------------------------------------------------------------
library(factoextra) #For plotting Clusters
#Deciding Number of Clusters

fviz_nbclust(x[,], kmeans, method = "wss")
kc<-kmeans(x[,],3)
kc

#Plotting Clusters
fviz_cluster(kc, data = x[,],
             palette = c("#0000FF", "#00FF00" ,"#FF0000"), #,"#000000"), 
             geom = "point",
             ellipse.type = "convex", 
             ggtheme = theme_bw())


fviz_nbclust(x,kmeans,method = "silhouette")



