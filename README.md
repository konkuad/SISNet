# EEGWaveNet (Multi-scale seizure prediction)
Seizure Classifier with multi-level convolution inspired by FBCSP

Literally every code is in EEGWaveNet.py

class Functions
This class has a lot of functions
1. build(n_chans,n_classes) => inputs the number of classes and channels you want to adjust in your task as integer
2. regular_train(Model,X_train,y_train,X_val,y_val,n_epochs,batch_size,learning_rate,patience,n_classes) just input your train and val data and hyperparams, it will output model and history.
4. evaluate(Model,X_test,y_test) just input your test data, it will return the performance score.
5. clustering(embeddings,label,n_classes) performs t-SNE dimensionality reduction of given test data and label.

-TBA-
