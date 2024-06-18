from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import   train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import os

# Change the working directory to the dataset location
os.chdir(r"D:\Datasets")

# Load the chemical process dataset
chem = pd.read_csv("ChemicalProcess.csv")

# Check for missing values in the dataset
print(chem.isnull().sum())

# Separate the target variable (Yield) from the input features
y = chem['Yield']
x = chem.drop(['Yield'],axis=1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state = 24)

# Create a SimpleImputer to handle missing values
imp = SimpleImputer(strategy='median').set_output(transform ="pandas")

# Impute the missing values in the training set
x_trn_imp = imp.fit_transform(x_train)

# Create and train the linear regression model
lr = LinearRegression()
lr.fit(x_trn_imp, y_train)

# Impute the missing values in the test set
x_tst_imp = imp.transform(x_test)

# Predict on the test set and calculate the R-squared score
y_pred = lr.predict(x_tst_imp)
print(y_pred)
print(r2_score(y_test, y_pred))
