# K-NearestNeighbors
K-Nearest Neighbors for classification and regression implemented from scratch.
  -Hyperparameter tuning was done to determine the best value of K that yields the lowest test error.
  -Tested on the make classification (random 2-class classification problem) and moons datasets from scikit-learn.
  -Accuracy was tested against the scikit-learn built-in KNeighborsClassifier and KNeighborsRegressor.

200 samples from each dataset were generated and randomly split into training and testing datasets. Training and testing error for varying values of the hyperparameter K as well as the classifications and regression visualizations can be found in the corresponding results folder.

All code is in the knn.py file. Simply run the file to reproduce the results.
