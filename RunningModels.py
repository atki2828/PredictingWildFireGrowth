#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import random as random
import KFold_CV_Class
from sklearn.metrics import classification_report,confusion_matrix
#%%
#This Code runs gridsearch on all potential models
### Grid Search on SVM
from sklearn.model_selection import train_test_split ,GridSearchCV
X=data1.drop('Growth',axis =1 )
Y= data1['Growth']

param_grid = {'C':[0.1,1,10,.001,0.5] ,'gamma':[0.5,0.1,0.01,.0001] ,'kernel':['rbf','sigmoid']}
from sklearn.svm import SVC
grid = GridSearchCV(SVC(),param_grid,verbose=0)
grid.fit(X,Y)
grid.best_params_


### Grid Search on RF
from sklearn.ensemble import RandomForestClassifier
param_grid = {'n_estimators':[1,50,75,100] ,'max_depth':[5,10,15] }
grid = GridSearchCV(RandomForestClassifier(),param_grid,verbose=0)
grid.fit(X,Y)
grid.best_params_

#### Grid Search KNN
from sklearn.neighbors import KNeighborsClassifier
param_grid = {'n_neighbors':list(range(40)) }
grid = GridSearchCV(KNeighborsClassifier(),param_grid,verbose=0)
grid.fit(X,Y)
grid.best_params_


#%%
#run Kfold_Class with Logistic Regression
from sklearn.linear_model import LogisticRegression
glm = LogisticRegression(solver ='newton-cg',penalty = 'none')
KF_glm = Kfold_CV_Classifier(k=10,model=glm,data=data2,response='Growth',name='Logistic Regression')
KF_glm.run_CV(seed=28)
KF_glm.plot_metric('0','MisClass')
KF_glm.mean_mofit('0','MisClass')
KF_glm.plot_metric('0','Recall')
KF_glm.mean_mofit('0','Recall')
KF_glm.Sum_Confusion('0')



#%%
#run Kfold_Class with Random Forest
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators= 50 ,max_depth=5)
KF_RF = Kfold_CV_Classifier(k=10,model=RF,data=data1,response='Growth',name='Random Forest')
KF_RF.run_CV(seed=28)
KF_RF.plot_metric('0','MisClass')
KF_RF.mean_mofit('0','MisClass')
KF_RF.plot_metric('0','Recall')
KF_RF.mean_mofit('0','Recall')
KF_RF.Sum_Confusion('0')

#%%
#run Kfold_Class with KNN
from sklearn.neighbors import KNeighborsClassifier
KNN =KNeighborsClassifier(26)
KF_KNN = Kfold_CV_Classifier(k=10,model=KNN,data=data1,response='Growth',name='KNN')
KF_KNN.run_CV(seed=28)
KF_KNN.plot_metric('0','MisClass')
KF_KNN.mean_mofit('0','MisClass')
KF_KNN.plot_metric('0','Recall')
KF_KNN.mean_mofit('0','Recall')
KF_KNN.Sum_Confusion('0')

#%%
#Run KfolcClass for SVM
from sklearn.svm import SVC
SVM = SVC(C=1,gamma=0.1 , kernel='sigmoid')
SVM_KNN = Kfold_CV_Classifier(k=10,model=SVM,data=data1,response='Growth',name='SVM')
SVM_KNN.run_CV(seed=28)
SVM_KNN.plot_metric('0','MisClass')
SVM_KNN.mean_mofit('0','MisClass')
SVM_KNN.plot_metric('0','Recall')
SVM_KNN.mean_mofit('0','Recall')
SVM_KNN.Sum_Confusion('0')

#%%
#Run KfolcClass for SVM With increase Cost budget
from sklearn.svm import SVC
SVM = SVC(C=200,gamma=0.1 , kernel='rbf')
SVM_KNN = Kfold_CV_Classifier(k=10,model=SVM,data=data1,response='Growth',name='SVM')
SVM_KNN.run_CV(seed=28)
SVM_KNN.plot_metric('0','MisClass')
SVM_KNN.mean_mofit('0','MisClass')
SVM_KNN.plot_metric('0','Recall')
SVM_KNN.mean_mofit('0','Recall')
SVM_KNN.Sum_Confusion('0')