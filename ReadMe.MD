  
## Predicting Wildfire Growth Project Overview
This project is an exercise in Binary Classification as it aims to compare popular classifiers on the Alberta fire data set. In this project I first perform some EDA to determine which variables may show predictive power in determining if a wildfire will grow from its initial report or not. Then I develop a K-folds Classification tool that is used to process and assess each binary classifiers performance. I then run KNN, SVM, RandomForest and Logistic Regression through the aforementioned tool and evaluate their performances in terms of accuracy and recall.

## Predicting Wildfire Growth Files
* Readme
* journal.pone.01898.pdf which is a paper tilted "Survival analysis and classification methods
for forest fire size"
* WildFireEDA.py contains code for EDA
* KFold_CV_Class.py contains code for building KFold Class to be used as tool for model comparison
* RunningModels.py contains code for tuning model params with grid search and model comparison
* Predicting Wild Fire Growth Analysis contains PDF of Jupyter notebook running code with written analysis of output
* Predicting_Wild_Fire_Growth.ipynb

## Technolody Used
* python
* pandas
* seaborn
* matplotlib
* scipy
* sklearn

## Project Takeaways
In original model runs we saw all models perform equally well in terms of pure accuracy. Each model had an average misclassification rate of roughly 0.25 and were all certainly with in the error of each other. However, if we take into account Recall and decide the goal of our models should be to determine as many of the fires that will grow as possible then the best model would be SVM with radial kernel and increased cost function that was run in the discussion of analysis section. If the purpose was pure classification accuracy, I would say to use logistic regression since it had the best score point score and was also what the authors of the paper found to be best for accuracy.