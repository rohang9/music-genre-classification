# music-genre-classification
This repository has some codes and models which can be used to do Genre Classification for Music. GTZAN dataset has been used to train the models


## List of Models Tried Out
* Traditional ML models
* Simple Neural Network by taking mean and std deviation of the features
* CNN on Spectogram Images
* LSTM on 1D Song Array
* LSTM on 1D Songs Array with Attention
* Time Distributed Dense followed with LSTM


### Accuracies Come As Follows

|**Model**|**Training Accuracy**|**Cross Validation Accuracy**|
|---------|------------|-------------|
|Logistic Regression|0.64|0.58|
|SVM|0.72|0.64|
|Random Forest|0.86|0.58|
|Simple Neural Netwerk|0.78|0.70|
