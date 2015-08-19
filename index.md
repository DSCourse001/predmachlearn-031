---
title       : Practical Machine Learning
subtitle    : This presentation is Course project for Practical Machine Learning class.
author      : DSCourse001 User (predmachlearn-031 class)
job         : 
framework   : io2012        # {io2012, html5slides, shower, dzslides, ...}
highlighter : highlight.js  # {highlight.js, prettify, highlight}
hitheme     : tomorrow      # 
widgets     : []            # {mathjax, quiz, bootstrap}
mode        : selfcontained # {standalone, draft}
knit        : slidify::knit2slides
github:
  user: DSCourse001
  repo: predmachlearn-031
---

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively.
One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

[http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har)

---

## The Goal 



The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set.

After that we need to use prediction model to predict 20 different test cases.


Ok. What we will do?

---

## Action Plan

We need an action plan to complete that exercise.

 1. Obtain the Data.
 2. Clean the Data.
 3. Prepare Data for futher analysis.
 4. Try to use _K Nearest Neighbor_ or _KNN_ method to predict.
 5. Try to use _Random Forest_ method to predict.
 6. Choose more accurate method.
 7. Use it to predict 20 different test cases.

---

### Obtaining the Data

The training data for this project are available here: 
[https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

The test data are available here (20 different test cases): 
[https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

Reading data using R.

```r
require(data.table)
Data<- fread("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
             heade=T,na.strings=c("NA","N/A","","#DIV/0!"))
tData<- fread("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
              heade=T,na.strings=c("NA","N/A","","#DIV/0!"))
```

---

## Clean the Data 1 of 2 

 * Converting all data to numeric

```r
Data<- cbind(
        Data[,!grepl("_(belt|arm|dumbbell|forearm)",names(Data)),with=F],
              apply(
                Data[,grepl("_(belt|arm|dumbbell|forearm)",names(Data)),with=F]
                ,2,"as.numeric"))
```

 * Removing NA values

```r
Data<-Data[,apply(Data,2,function(x) !any(is.na(x))),with=F]
```

---

## Clean the Data 2 of 2 

 * Remove Unused Columns

```r
Data<-Data[,-(1:7),with=F]
```

 * Convert _casse_ column values to factor

```r
Data$classe<-factor(Data$classe)
```

 * Finally we have Data Frame

```r
dim(Data)
```

```
## [1] 19622    53
```

---

## Prepare Data for futher analysis

Now we can form two Data Sets: 
 * for *training* purposes (60% of data)
 * for *testing* purposes (40% of data) to perform Cross Validation.

_Note:_ Don't forget to set seed value.


```r
require(caret)
set.seed(25)
index = createDataPartition(Data$classe, p = .6)[[1]]
training = Data[ index,]
testing = Data[-index,]
```

---

## K Nearest Neighbor 1 of 2

The Idea.

_"A K nearest neighbor classifies new samples by first finding the K closest
samples in the training set and determines the predicted value based on the known outcomes of the nearest neighbors"_

[http://arxiv.org/pdf/1405.6974v1.pdf](http://arxiv.org/pdf/1405.6974v1.pdf)

[https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

---

## K Nearest Neighbor 2 of 2

R code:

```r
# Pre-process the data
procData <- preProcess(training[,-1,with=F],method=c("knnImpute"))

# Getting Nearest
trainingKNN<-predict(procData,training[,-1,with=F])
testingKNN<-predict(procData,testing[,-1,with=F])

# Model Data Behaviour
model1 <- train(training$classe ~.,data=trainingKNN, method="knn")

# Getting Results
results1 <- confusionMatrix(testing$classe, predict(model1,testingKNN))
```

---

## Random Forest 1 of 2

The Idea:

_"The training algorithm for random forests applies the general technique of bootstrap aggregating, or bagging, to tree learners."_

_"Random forests differ in only one way from this general scheme: they use a modified tree learning algorithm that selects, at each candidate split in the learning process, a random subset of the features."_

[https://en.wikipedia.org/wiki/Random_forest](https://en.wikipedia.org/wiki/Random_forest)

---

## Random Forest 2 of 2

R code:



```r
# Setting Parameters
trainControl2<-trainControl(method="cv", number=3, allowParallel=T)

# Model Data Behaviour
model2<- train(classe~.,data=training,method="rf",trControl=trainControl2)

# Getting Results
results2 <- confusionMatrix(testing$classe,predict(model2,testing))
```

---

## Choose more accurate method 1 of 2 

Results from Cross Validation


|Method             | Accuracy|
|:------------------|--------:|
|K Nearest Neighbor |    95.55|
|Random Forest      |    99.13|

*K Nearest Neighbor* Confusion Matrix

```
##           Reference
## Prediction    A    B    C    D    E
##          A 2199   18    8    5    2
##          B   55 1409   46    2    6
##          C    6   33 1300   24    5
##          D    1    0   73 1204    8
##          E    3   21   18   15 1385
```

---

## Choose more accurate method 2 of 2

*Random Forest* Confusion Matrix

```
##           Reference
## Prediction    A    B    C    D    E
##          A 2229    1    2    0    0
##          B    7 1501   10    0    0
##          C    0   10 1354    4    0
##          D    0    1   24 1258    3
##          E    0    1    3    2 1436
```

Now We see that *Random Forest* is more accurate. We will use it for prediction.

---

## Predict 20 different test cases

Prediction using Random Forest.


```r
predict(model2,tData)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

---

## The End

<img class=center src=http://www.happyologist.co.uk/wp-content/uploads/happy.jpeg height=350>

Thank you for reading.

Happy image was grabbed from the URL [http://www.happyologist.co.uk/wp-content/uploads/happy.jpeg](http://www.happyologist.co.uk/wp-content/uploads/happy.jpeg).

---
