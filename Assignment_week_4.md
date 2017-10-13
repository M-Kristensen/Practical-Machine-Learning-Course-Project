### Aim

The aim of this project is to create a report describing how I built a
classification model of human activity recognition, how I used
cross-validation, what I think the expected out of sample error is, and
why I made the choices I did.

### Background

*The data and the background info text is from
<http://groupware.les.inf.puc-rio.br/har>.*

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement – a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behaviour, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website here:
<http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight
Lifting Exercise Dataset).

#### Activity Model

Six young health participants were asked to perform one set of 10
repetitions of the Unilateral Dumbbell Biceps Curl in five different
fashions: exactly according to the specification (**Class A**), throwing
the elbows to the front (**Class B**), lifting the dumbbell only halfway
(**Class C**), lowering the dumbbell only halfway (**Class D**) and
throwing the hips to the front (**Class E**).

Class A corresponds to the specified execution of the exercise, while
the other 4 classes correspond to common mistakes. Participants were
supervised by an experienced weight lifter to make sure the execution
complied to the manner they were supposed to simulate. The exercises
were performed by six male participants aged between 20-28 years, with
little weight lifting experience.

### Data

The data used for this project are available using the following links:

Training data:
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

Test data:
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

#### Setup - Getting and Preparing the Data Sets

    packages <- c("caret", "randomForest", "parallel","doParallel", "knitr")
    sapply(packages, require, character.only = TRUE, quietly = TRUE)

Retrieve and load the data sets.

    urlTrain <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    urlTest <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

    training <- read.csv(url(urlTrain), na.strings=c("NA","#DIV/0!",""))
    testing <- read.csv(url(urlTest), na.strings=c("NA","#DIV/0!",""))

Let us check the dimensions of the data, to get an idea of what we are
dealing with.

    dim(training)

    ## [1] 19622   160

We find that the training data contains 19622 rows of data, which should
be plenty to perform cross-validation.

#### Preparing the Data Sets

By viewing the data we notice that some of the columns in the data sets
contain irrelevent information (the first seven coloumns as it turns
out: "X", "user\_name", "raw\_timestamp\_part\_1",
"raw\_timestamp\_part\_2", "cvtd\_timestamp", "new\_window",
"num\_window"). Also, many of the variables appears to hold only NA
values.

To avoid unnecessary clutter we start off by removing these columns.

    # Update data sets to exclude variables with only NA values
    training <- training[,colSums(is.na(training)) == 0]
    testing <- testing[,colSums(is.na(testing)) == 0]

    # Remove variables that are irrelevant to the prediction
    training <- training[, -c(1:7)]
    testing <- testing[, -c(1:7)]

As a final attempt to clean the data, we remove near zero variance
variables (if there are any).

    # Remove near zero variance variables
    training <- training[, nearZeroVar(training, saveMetrics = T)$nzv == F]
    testing <- testing[, nearZeroVar(testing, saveMetrics = T)$nzv == F]

### A Note on Model Selection

I have applied the train function in caret using several different
methods, and five-fold cross-validation. The result with respect to
accuracy is as follows:

1.  "**rf**" (random forest) accuracy was 0.9946982
2.  "**rpart**" (recursive partitioning and regression trees) accuracy
    was 0.7402121
3.  "**gbm**" (stochastic gradient boosting) accuracy was 0.9647227
4.  "**xgbTree**" (extreme gradient boosting) accuracy was 0.9963295

I also tried improving the accuracy by combining the rf model and the
xgbTree model, which result in an accuracy of 0.9973491. Using this
approach comes at a price, however, as cross-validation with the caret
train function is quite computationally intensive.

As a result of these considerations, I settled on simply using
randomForest, with only a single cross-validation sample. This approach
is **much** less power hungry, and the accuracy is nearly as good as the
best of the above mentioned approaches, and I find no need to improve
the accuracy further.

#### Cross-Validation

We split the training data into two groups (p = 0.75), and use the
larger group for training, and the smaller for cross-validation.

    set.seed(42)

    inTrain <- createDataPartition(training$classe, p=0.75, list=F)

    trainingCV <- training[inTrain,]
    testingCV <- training[-inTrain,]

To speed up the analysis, we now take advantages of multiple cores.

    clustFit <- makeCluster(detectCores() - 1)
    registerDoParallel(clustFit)

    model_RF <- randomForest(classe ~ .,
                             data=trainingCV,
                             method="class",
                             ntree = 1000)

    stopCluster(clustFit)

### Evaluating the model

To estimate the accuracy, and the out-of-sample error, we test the model
against the validation data.

    prediction_RF <- predict(model_RF, testingCV)
    confusionMatrix(prediction_RF, testingCV$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1395    3    0    0    0
    ##          B    0  946    5    0    0
    ##          C    0    0  850    9    0
    ##          D    0    0    0  795    1
    ##          E    0    0    0    0  900
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9963          
    ##                  95% CI : (0.9942, 0.9978)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9954          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9968   0.9942   0.9888   0.9989
    ## Specificity            0.9991   0.9987   0.9978   0.9998   1.0000
    ## Pos Pred Value         0.9979   0.9947   0.9895   0.9987   1.0000
    ## Neg Pred Value         1.0000   0.9992   0.9988   0.9978   0.9998
    ## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Rate         0.2845   0.1929   0.1733   0.1621   0.1835
    ## Detection Prevalence   0.2851   0.1939   0.1752   0.1623   0.1835
    ## Balanced Accuracy      0.9996   0.9978   0.9960   0.9943   0.9994

From the confusionmatrix, we get an **accurary of 99.63%**. This
corresponds to an **out-of-sample error of only 0.37%**.

### Final Prediction

The result of predicting the activity from the test data using model\_RF
is:

    FinalPrediction <- predict(model_RF, testing)
    kable(t(data.frame(FinalPrediction)))

<table>
<thead>
<tr class="header">
<th></th>
<th align="left">1</th>
<th align="left">2</th>
<th align="left">3</th>
<th align="left">4</th>
<th align="left">5</th>
<th align="left">6</th>
<th align="left">7</th>
<th align="left">8</th>
<th align="left">9</th>
<th align="left">10</th>
<th align="left">11</th>
<th align="left">12</th>
<th align="left">13</th>
<th align="left">14</th>
<th align="left">15</th>
<th align="left">16</th>
<th align="left">17</th>
<th align="left">18</th>
<th align="left">19</th>
<th align="left">20</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>FinalPrediction</td>
<td align="left">B</td>
<td align="left">A</td>
<td align="left">B</td>
<td align="left">A</td>
<td align="left">A</td>
<td align="left">E</td>
<td align="left">D</td>
<td align="left">B</td>
<td align="left">A</td>
<td align="left">A</td>
<td align="left">B</td>
<td align="left">C</td>
<td align="left">B</td>
<td align="left">A</td>
<td align="left">E</td>
<td align="left">E</td>
<td align="left">A</td>
<td align="left">B</td>
<td align="left">B</td>
<td align="left">B</td>
</tr>
</tbody>
</table>
