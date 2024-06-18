from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import   train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import os


# Change the working directory to the dataset location
os.chdir(r"D:\Datasets")

# Load the concrete dataset
conc_data = pd.read_csv("Concrete_data.csv")

# Separate the target variable (Strength) from the input features
y = conc_data['Strength']
x = conc_data.drop(['Strength'],axis=1)


# Apply polynomial feature transformation with degree 3
poly =PolynomialFeatures(degree = 3)
x_poly = poly.fit_transform(x)

print(poly.get_feature_names_out())

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size=0.3,random_state = 24)

# Create and train the linear regression model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Print the coefficients and intercept of the model
print(lr.coef_)
print(lr.intercept_)

# Predict on the test set and calculate the R-squared score
y_pred = lr.predict(x_test)
print(y_pred)
print(r2_score(y_test,y_pred))

#When we compare between degree 2 and degree 3 , we come to a conclusion that model with degree 3 is best

#-------------------------------new dataset (predict value)(overfitting)-------------------------------------------

#Predicting on the un-labelled data is called Inferencing 

# Load the new dataset and apply the same polynomial feature transformation
tst_conc = pd.read_csv('testConcrete.csv')
tst_poly = poly.fit_transform(tst_conc)

# Use the trained model to predict the concrete strength for the new data
pred_str = lr.predict(tst_poly)



















