
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import   train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import os
from sklearn.preprocessing import PolynomialFeatures
os.chdir(r"D:\Datasets")


chem  = pd.read_csv("ChemicalProcess.csv")
print(chem.isnull().sum())

y = chem['Yield']
x = chem.drop(['Yield'],axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state = 24)

imp = SimpleImputer(strategy='median').set_output(transform ="pandas")

x_trn_imp = imp.fit_transform(x_train)
lr = LinearRegression()
lr.fit(x_trn_imp, y_train)

##Test set operations
x_tst_imp = imp.transform(x_test)
y_pred = lr.predict(x_tst_imp)
print(r2_score(y_test, y_pred))