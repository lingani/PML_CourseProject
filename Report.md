# Practial Machine Learning: Course Project
Elliot Waingold

## Introduction

This report describes the selection & training of a machine learning algorithm to predict the manner in which an exercise was performed based on sensor readings.  We will use data from a [study](http://groupware.les.inf.puc-rio.br/har) in which six participants were asked to perform barbell lifts in five different ways (one correctly and the others incorrectly).  Specifically, the input data for the prediction are the readings from accelerometers placed on each participant's arm, forearm, & dumbell.  The outcome is classification into one of the five ways in which the exercise was performed.

## Data preparation

We begin preparing the data by fetching it from the download point.


```r
# acquire data from specific location
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "pml-training.csv")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "pml-testing.csv")
```

We proceed to read the provided training data with appropriate treatment of missing values.


```r
setwd("~/R/Coursera/Practical Machine Learning")
data <- read.csv("pml-training.csv", na.strings = c("NA", "#DIV/0!"))
dim(data)
```

```
## [1] 19622   160
```

Based on our understanding of the domain, we choose to drop variables that contain missing values and those that represent sample identification values (as opposed to sensor readings).


```r
# mask out variables that have NA values
colMask <- as.vector(colSums(is.na(data)) == 0)
# mask out variables that don't represent sensor readings
colMask[1:7] <- FALSE
data <- data[, colMask]
names(data)
```

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"     "classe"
```

Finally, we partition the data into training & testing sets using a 70:30 split.  The testing set will be used to estimate out-of-band accuracy for the final trained prediction model.


```r
# create training & testing sets
library(caret)
inTrain <- createDataPartition(data$classe, p = 0.7, list = F)
training <- data[inTrain,]
testing <- data[-inTrain,]
dim(training); dim(testing)
```

```
## [1] 13737    53
```

```
## [1] 5885   53
```

## Algorithm selection

This prediction problem involves predominantly numeric covariates and a purely categorical outcome.  There are a number of machine learning algorithms suited to such a task.  We choose to evaluate five such candidate algorithms: linear discriminant analysis (LDA), recursive partitioning (RP), random forests (RF), generalized boosted regression modeling (GBM), & naive Bayes (NB).  For each candidate algorithm, we evaluate performance with and without preprocessing via principal components analysis (PCA).

For each candidate algorithm, we perform 2-fold cross-validation over the 70% training set.  We use the average OOB accuracy to judge suitability of the algorithm to this application.


```r
# create partitions for 2-fold cross-validation
inPart1 <- createDataPartition(training$classe, p = 0.5, list = F)
trainSets <- list(training[inPart1,], training[-inPart1,])
sapply(trainSets, dim)
```

```
##      [,1] [,2]
## [1,] 6869 6868
## [2,]   53   53
```

```r
# define a method that performs 2-fold cross-validation and returns the average accuracy
evalMethod <- function(method, preProcess = NULL, ...) {
    modFits <- lapply(trainSets, function(data) 
        train(classe ~ ., method = method, preProcess = preProcess, data = data, ...))
    confMats <- lapply(1:2, function(i) { 
        newdata <- trainSets[[3 - i]]
        confusionMatrix(predict(modFits[[i]], newdata = newdata), newdata$classe)
    })
    accuracy <- mean(sapply(confMats, function(cm) cm$overall["Accuracy"]))
    accuracy
}
```


```r
evalMethod("lda", "pca"); evalMethod("lda")
evalMethod("rpart", "pca"); evalMethod("rpart")
evalMethod("rf", "pca"); evalMethod("rf")
evalMethod("gbm", "pca", verbose = F); evalMethod("gbm", verbose = F)
evalMethod("nb", "pca"); evalMethod("nb")
```

The results, summarized in the table below, suggest that RF without PCA preprocessing would yield the most accurate model.  We seriously considered using RF with PCA given its significantly lower training time.  However, we decided that the training time of RF without PCA was not prohibitice for this application.


|Algorithm |PCA? |Est. Accuracy |
|:---------|:----|:-------------|
|LDA       |yes  |53.8%         |
|LDA       |no   |69.9%         |
|RP        |yes  |39.2%         |
|RP        |no   |53.1%         |
|RF        |yes  |94.6%         |
|RF        |no   |98.4%         |
|GBM       |yes  |80.4%         |
|GBM       |no   |95.3%         |
|NB        |yes  |64.4%         |
|NB        |no   |73.1%         |

## Model training

Having selected an algorithm (RF without PCA), we now turn to training a model over the full 70% training set.


```r
modFit <- train(classe ~ ., method = "rf", data = training)
```

We now conduct prediction over the 30% testing set in order to arrive at a final estimate of OOB accuracy.


```r
confusionMatrix(predict(modFit, newdata = testing), testing$classe)
```
```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1673    6    0    0    0
         B    0 1131    0    0    0
         C    0    2 1018   12    1
         D    0    0    8  951    1
         E    1    0    0    1 1080

Overall Statistics
                                          
               Accuracy : 0.9946          
                 95% CI : (0.9923, 0.9963)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9931          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9994   0.9930   0.9922   0.9865   0.9982
Specificity            0.9986   1.0000   0.9969   0.9982   0.9996
Pos Pred Value         0.9964   1.0000   0.9855   0.9906   0.9982
Neg Pred Value         0.9998   0.9983   0.9984   0.9974   0.9996
Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
Detection Rate         0.2843   0.1922   0.1730   0.1616   0.1835
Detection Prevalence   0.2853   0.1922   0.1755   0.1631   0.1839
Balanced Accuracy      0.9990   0.9965   0.9946   0.9923   0.9989
```

These results suggest excellent OOB accuracy of ~99.5% (i.e., OOB error rate of ~0.5%) within a tight 95% confidence interval.

## Prediction results

Finally, we use the trained model to predict the outcomes of the supplied testing data.


```r
setwd("~/R/Coursera/Practical Machine Learning")
data <- read.csv("pml-testing.csv", na.strings = c("NA", "#DIV/0!"))
dim(data)
```

```
## [1]  20 160
```

```r
predict(modFit, newdata = data)
```
```
 [1] B A B A A E D B A A B C B A E E A B B B
Levels: A B C D E
```

## Conclusion

A random forest model (without PCA preprocessing) was trained to predict the manner in which barbell lifts were performed based on accelerometer readings.  The model exhibits very high OOB accuracy and correctly predicted outcomes for all 20 samples in the provided testing data.
