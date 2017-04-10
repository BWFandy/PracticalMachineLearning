---
title: "Practical Machine Learning - Course Project"
author: "AS"
date: "April 10, 2017"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).

### Data

The training data for this project are available here:  
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>  

The test data are available here:  
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>  

The data for this project come from this source: <http://groupware.les.inf.puc-rio.br/har>.  

## Executive Summary:

Using the data provided above, and the materials from the "Practical Machine Learning" lectures, this report summarizes the creation of a machine learning algorithm to predict the correct class of exercise from the data described above.  

Topics covered in this report refer directly to the criteria for this project, namely:  
- How the model was built  
- How cross-validation was used  
- What the expected out-of-sample error is  
- Why certain choices were made  

Each of these topics is covered in the report below, as well as sections on **data clean-up, feature selection, model parameters, and model evaluation.**  

## Reading in & data, initial evaluation and clean-up:  

The following R packages were used in this analysis:  
```{r, include = FALSE}
library(caret)
library(Hmisc)
library(dplyr)
library(caret)
library(corrplot)
library(e1071)
```

### Read in the data:  
'na.strings' argument was used to cleanly-identify missing data when data was read in so that it could later be deleted easily:    
```{r}
training <- read.csv("pml-training.csv", header=TRUE, sep=",", na.strings = c("NA", ""))
testing <- read.csv("pml-testing.csv", header=TRUE, sep=",", na.strings = c("NA", ""))
```

### Initial data evaluation and exploration:  
A general evaluation of the training data was performed. During this evaluation, it was found that most of the derived data points were actually missing from the data set, particularly when **window_new** = 'yes':   
```{r, include = FALSE}
summary(training)
describe(training)
```

```{r}
table(training$user_name)
table(training$classe)
```


Due to the broad scope of missing data, it was decided to omit columns with majority missing data from further analysis. A function was created for this purpose:  

```{r}
# Created a Function to identify columns with lots of missing data:
miss <- function(x) {
        as.vector(apply(x, 2, function(x) length(which(is.na(x)))))
}
```

```{r}
# Put the identifier into a new vector and called it within the 'for' loop below (19216 represents the number of missing cases in each variable to be omitted):
countNA <- miss(training)
countNA

omit <- c()
for (cnt in 1:ncol(training)) {
        if (countNA[cnt] == 19216) {
                omit <- c(omit, colnames(training)[cnt])
        }
}
```

```{r}
# Omit columns with lots of NA values from training and testing sets:
training <- training[, !(names(training) %in% omit)]
testing <- testing[, !(names(testing) %in% omit)]

# Verify remaining columns are the same in both training and testing sets (except training$classe and testing$problem_id):
colnames(training)
colnames(testing)
```


### Evaluating the remaining (non-missing) variables:  
The remaining variables were then evaluated for predictive potential relating to this classification problem:  

```{r}
# Identify variables with Near Zero Variance:
nzv <- nearZeroVar(training, saveMetrics=TRUE)
nzv
```

Only 'new_window' variable is listed as having near zero variance, but this - along with the timestamp variables - will not be used for prediction due to their irrelevance to the classification problem. They are omitted below:  
```{r}
table(training$new_window)
# training$new_window is listed as near zero variance, but will be omitted:
training <- select(training, 8:60)
testing <- select(testing, 8:60)
```

Further evaluation was performed on the remaining variables to analyze normalcy and correlations between variables:  

```{r}
# In order to look at our data in bulk, we should group all numerics together using 'select' from <dplyr>:
dat.matrix <- as.matrix(select(training, 1:52))

# we can now look at normality stats for all numeric vars:
skew <- apply(dat.matrix, 2, skewness)
skew
summary(skew)
hist(skew)

kurt <- apply(dat.matrix, 2, kurtosis)
kurt
summary(kurt)
hist(kurt)
```
  
Some of these variables exhibit skewness and kurtosis, which may impact our analysis depending on the model's robustness to non-normal data. Therefore, some pre-processing may be necessary to make the data more normal & improve overall model performance.   

```{r, include = FALSE}
# correlation matrix:
correlations <- cor(dat.matrix)
correlations
```

```{r}
# plotting correlations:
corrplot(correlations, order = "hclust")

# identify highly-correlated variables for potential deletion:
highCor <- findCorrelation(correlations, cutoff = .75) # returns column numbers
highCor
```
  
There are also several variables which exhibit significant between-variable correlation and should be omitted to simplify our analysis:  

```{r}
# Omit columns of highly-correlated variables from training and testing sets:
training <- training[, -highCor]
testing <- testing[, -highCor]
```

## Splitting Initial Training Data into Training and Validation Sets:  
Now that the data sets have been evaluated and cleaned-up, we can prepare for modeling.  
Based on the extremely small size of the testing set (n = 20), the training data was further broken down into "train1" and "val1" training and validation sets:  
```{r}
set.seed(342)
train_sample <- createDataPartition(y = training$classe,
                                  p = .6,
                                  list = FALSE)

train1 <- training[train_sample,]
val1 <- training[-train_sample,]
```

## Establishing a Baseline Model for comparison purposes:  
A true **random guess baseline model** would simply guess each of the 5 classes approx. 3924 times (1/5 of the time)
```{r}
table(training$classe)
prop.table(table(training$classe))
```

### Baseline #1:  
We can improve on the "random guess" model by constructing a simple decision tree model on 'train1' using all default settings, with no pre-processing or cross-validation:  
```{r}
set.seed(342)
baseMod <- train(classe ~ .,
                 data = train1,
                 method = "rpart")
baseMod
baseMod$finalModel
```
The Baseline Model #1, with its Accuracy of 55%, doesn't do much better than random guessing. We should expect subsequent models to improve on this initial accuracy.  

### Candidate Model #1: Decision Tree with some pre-processing:  
The model below builds on the simple baseline model by adding some pre-processing in order to deal with the non-normal characteristics of the data seen during the Exploration phase:  
```{r}
set.seed(342)
can1Mod <- train(classe ~ .,
                 data = train1,
                 preProcess=c("center", "scale", "BoxCox"),
                 method = "rpart")
can1Mod
```
Unfortunately, this model exhibits nearly the same prediction accuracy (54%), which is similarly disappointing.  


### Candidate 2: Decision Tree with pre-processing, and cross-validation:  
The model below includes 2 repeats of 10-fold cross-validation in order to build on candidate model 1.  
```{r}
modelLookup("rpart")
set.seed(342)

# Set resampling method using 'trainControl' from Caret pkg:
train_control <- trainControl(method = "repeatedcv", # type of resampling
                              number = 10,           # number of folds
                              repeats = 2)           # repeats of whole process

can2Mod <- train(classe ~ .,
                 data = train1,
                 preProcess = c("center", "scale", "BoxCox"),
                 trControl = train_control,
                 method = "rpart")

can2Mod
```
The addition of cross-validation showed some minor improvement over previous models (model accuracy of 55.6% versus previous best of 55.3%).  
However, this is still only a small improvement over random guessing. Therefore, the more advanced *Random Forest* machine learning algorithm will be utilized in subsequent tests in order to obtain a more satisfactory classification rate.  

### Candidate 3: Random Forest model with pre-processing and cross-validation:  
Based on previous efforts (including the candidate models above), the lecture notes, and the Skewness/Kurtosis characteristics of the data, it seems that both pre-processing and cross-validation will lead to improved model performance:  
```{r, include = FALSE}
modelLookup("rf")
set.seed(342)

can3Mod <- train(classe ~ .,
                 data = train1,
                 preProcess = c("center", "scale", "BoxCox"),
                 trControl = train_control,
                 method = "rf")
```

```{r}
can3Mod
can3Mod$finalModel
```
The random forest model shows much better accuracy (98.7%), as well as confusion matrix and OOB error estimates which are much better than previous attempts. This model will be used as our "Final Model" & will be measured against the validation and test data sets.  

## Using the Final Model to Predict on Validation Set:  
Due to the high accuracy of the random forest model (candidate #3), predictions will be made to the validation set for confirmation of final model selection:  

```{r}
### Predict on the validation set:
preds <- predict(can3Mod, newdata=val1)
confusionMatrix(preds, val1$classe)
```

## Using Predictions on Validation Set to Evaluate Out-Of-Sample Error Rate:  
Now that predictions have been made on new data, we can estimate our out-of-sample error rate as follows:  
```{r}
# Out-of-sample error rate: The error rate on a new data set - ie, from the validation set:
errorRate <- 1 - .9911
errorRate
```

## Final predictions on the Testing set:  
To fulfill the assignment's final requirement that predictions be made on the 20 observations in the Testing set, the final model was used to predict those values and submitted as Week 4's Quiz:  
```{r}
finalPreds <- predict(can3Mod, newdata=testing)
finalPreds
```

## Summary:  
- We began with a thorough understanding of our data and the question at hand (ie, the project criteria)  
- Training and Testing data were read in, evaluated, explored, and cleaned as needed to provide the best predictions possible  
- The original data was simplified based on a few standard methods  
- Several candidate models were built and evaluated using our understanding of the data and modeling techniques  
- A final model was selected based on accuracy characteristics, overall classification, and out-of-sample error estimates  
- The final model was applied to the validation data set for confirmation, and ultimately to the Testing set for final predictions which will be submitted as Week 4's Quiz answers in order to fulfill the requirements of this Project and Course.  
