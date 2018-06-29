##### Polynomial regression
#####  

import matplotlib.pyplot as plt
import pandas as pd

## load the data and separate the columns as X and y
data_0 = pd.read_csv('data_0.csv')
X = data_0.iloc[:,0:1].values
y = data_0.iloc[:,1].values


### splitting of the data set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 0)
####


#### polynomial regression
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X_train,y_train)


#### prediction 
y_linear  =  linear_regression.predict(X_train)

#### plot
plt.scatter(X_train, y_train, color ='blue')
plt.plot(X_train,y_linear, color = 'orange')
plt.title('linear regression fit of the model')
plt.xlabel('Change in water level')
plt.ylabel('Water flowing out of the Dam')
plt.show()




### Results for the polynomial features analysis
from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree = 2)
X_polynomial = polynomial_regression.fit_transform(X_train)
### polynomial_regression.fit(X_polynomial,y_train)
#################
linear_regression_0 = LinearRegression()
linear_regression_0.fit(X_polynomial, y_train)


y_polynomial =  linear_regression_0.predict(X_polynomial)
### plot

plt.scatter(X_train, y_train, color ='blue')
plt.scatter(X_train, y_polynomial, color = 'red')
plt.title('Polynomial model')
plt.xlabel('Change in water level')
plt.ylabel('Water flowing out of the Dam')
plt.show()

#### confusion matrix











