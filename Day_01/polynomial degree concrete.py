
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import   train_test_split
from sklearn.preprocessing import OneHotEncoder
import os
from sklearn.preprocessing import PolynomialFeatures
os.chdir(r"D:\Datasets")


conc_data = pd.read_csv("Concrete_data.csv")

y = conc_data['Strength']
x = conc_data.drop(['Strength'],axis=1)

from sklearn.preprocessing import PolynomialFeatures
poly =PolynomialFeatures(degree = 3)
x_poly = poly.fit_transform(x)
print(poly.get_feature_names_out())

x_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size=0.3,random_state = 24)


lr = LinearRegression()
lr.fit(x_train, y_train)
print(lr.coef_)
print(lr.intercept_)
y_pred = lr.predict(x_test)
print(r2_score(y_test,y_pred))
#print(y_pred)

#When we compare between degree 2 and degree 3 , we come to a conclusion that model with degree 3 is best

#-------------------------------new dataset (predict value)(overfitting)-------------------------------------------

#Predicting on the un-labelled data is called Inferencing 
tst_conc = pd.read_csv('testConcrete.csv')
tst_poly = poly.fit_transform(tst_conc)
pred_str = lr.predict(tst_poly)
























