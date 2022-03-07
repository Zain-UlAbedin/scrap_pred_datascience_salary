# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 12:55:07 2022

@author: Zain Khan
"""
# All required libraries will be at the top


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import pickle


import statsmodels.api as sm # will be used to train regression (you can also use sklearn regression as given below)

df = pd.read_csv('eda_data.csv')

"""
Tasks to do
- Choose relevant columns
- onehotcode categorical data/get dummy data
- create train test split dataset
- use multiple linear regression
- use lasso regression
- apply random forest
- test ensembles
- tune the models using gridsearchCV/RandomsearchCV(when you want the model to select combinations randomly)
"""
df_model = df[['avg_salary','Rating','Size','Type of ownership','Industry','Sector','Revenue','num_competitors','hourly','employer_provided',
             'job_state','same_state','age','python','spark','aws','excel ','job_simplified','seniority', 'desc_len']]




# here we get dummy data for all categorical data
df_dum = pd.get_dummies(df_model)

# now that we have numerical dataset lets create train test dataset


X = df_dum.drop('avg_salary', axis =1)
y = df_dum.avg_salary.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# now that we have train and test dataset we will build our models

X_sm = sm.add_constant(X)  # Linear Regression using statsmodel
model = sm.OLS(y, X_sm)
print(model.fit().summary())

# if you have noticed we have used the whole dataset on training regression model using statsmodel library, this
# time we will be using only train dataset to train the sklearn regression model

lm = LinearRegression()
lm.fit(X_train, y_train)

print("\n\n\nBelow given values are of cross validation, what is cross validation?\n-> Cross validation is basically \
 when we try to run the model on part of the train dataset and check it on other different part of the same dataset \
 e.g., train data set will be split in let say 3 categories in our case (bcz cv is 3), the model will be train on first 2 parts and validated on 2nd and then so on so forth till it has been test on all parts.\n \
 Below are the model validation score (meaning the model predicted score can be plus/minus the value.\n\n->")
                                      
                                       
                                      
cross_score = cross_val_score(lm, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)
print(cross_score)                       
print("Regression mean cross validation score: ", np.mean(cross_score))

# now that we have trained our regression lets use lasso regression which basically normalizes the data
lasso_r = Lasso()
lasso_r.fit(X_train, y_train)
print("Lasso Score: ", np.mean(cross_val_score(lasso_r, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)))

# it seems that Lasso didn't help much lets use different alpha values (alpha value by default is 1) and alpha value of 0 will make lasso regression simple regression

alpha_val = []
error_val = []

for alpha in range(1, 100):
    alpha_val.append(alpha/100)
    lassoR = Lasso(alpha=alpha/100)
    mean_cross_val_score =  np.mean(cross_val_score(lassoR, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))
    error_val.append(mean_cross_val_score)

plt.plot(alpha_val, error_val)  # this will show a line indicates what error value to expect on what alpha value
plt.show()


alpha_err = tuple(zip(alpha_val, error_val))
df_err = pd.DataFrame(alpha_err, columns=['alpha', 'error'])

print(df_err[df_err['error'] == max(df_err.error)])
# The value has been reduced by almost 1.5, meaning the alhpa value have helped though very slightly
# To clear, the negative value means that the predicted average salaryby our model can be off by plus minus 19 or less (19 means 19k)

# now that we have trained the above model we will train the last model which is random forest
# then we will try to tune the model and see if the model improve in performance.

rf = RandomForestRegressor()

print("Random Forest mean_cross val score: ", np.mean(cross_val_score(rf, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)))

# Oh, now Random forest has performed really well compared to the other models. Now that we have all models trained
# lets tune them and see if we can make them better. Because in real world the accurate the model the better.

# the parameters are for randomforest
parameters_rf = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}
gs = GridSearchCV(rf, parameters_rf, scoring='neg_mean_absolute_error', cv =3)
gs.fit(X_train, y_train)

print("Tuned model best score: ", gs.best_score_)
print("Tuned model best estimater: ", gs.best_estimator_)

# now that we have trained the model lets test all of the model on test data
# test ensembles

pred_lm = lm.predict(X_test)
pred_lasso = lasso_r.predict(X_test)
pred_rf = gs.best_estimator_.predict(X_test)

print("Value of LR:", mean_absolute_error(y_test, pred_lm))
print("Value of LassoR:", mean_absolute_error(y_test, pred_lasso))
print("Value of RF:", mean_absolute_error(y_test, pred_rf))

### Now that we have trained and test the models we know that Random forest performed better
### So to avoid training the model again and again we will save it to a file so we can just load it and use it
### on new data without training the whole model again.

import pickle
pickl = {'model': gs.best_estimator_}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ) )

file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

print(model.predict(np.array(list(X_test.iloc[1,:])).reshape(1,-1))[0])



