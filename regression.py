import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#archive\\data\\
df = pd.read_csv('GP1_0.6_marker.csv',sep=',')

x = []
y = []
var = 0
flag = 1
# print(df['L_FM1_x'])
for i in df['L_FM1_x']:
    if(var<(1.75/2)):
        if(var>=0.2):
            if(flag==1):
                y.append(i)
                x.append(var)
                flag = 0
            else:
                flag = 1;
        
        var += (0.005/2)
    else:
        break

x = np.array(x)
x = x.reshape(-1,1)

df1 = pd.read_csv('GP1_1.2_marker.csv',sep=',')
x1 = []
y1 = []

var = 0.2
# print(df['L_FM1_x'])
for i in df1['L_FM1_x']:
    if(var<(1.75/2)):
        y1.append(i+0.5)
        x1.append(var)
        var += (0.005)
    else:
        break

x1 = np.array(x1)
x1 = x1.reshape(-1,1)

x = x[:-1]
y = y[:-1]
# plt.plot(x1,y1)
# plt.show()

# regression_model = LinearRegression() 
# # Fit the data(train the model) 
# regression_model.fit(x, y) 

# print('Slope of the line is', regression_model.coef_) 

# print('Intercept value is', regression_model.intercept_) 
# Predict 

# y_predicted = regression_model.predict(x) 

# plt.scatter(x, y, s = 10) 

# plt.xlabel("$x$", fontsize = 18) 

# plt.ylabel("$y$", rotation = 0, fontsize = 18) 

# plt.title("data points") 

  
# # predicted values 

# plt.plot(x, y_predicted, color ='g') 

# plt.show()


poly_features = PolynomialFeatures(degree = 3, include_bias = False) 
x_poly = poly_features.fit_transform(x) 
lin_reg = LinearRegression() 
lin_reg.fit(x_poly, y) 
print('Coefficients of x are', lin_reg.coef_) 
print('Intercept is', lin_reg.intercept_) 

# x_new = np.linspace(-3, 4, 100).reshape(100, 1) 

x_new = x
x_new_poly = poly_features.transform(x_new) 
y_new = lin_reg.predict(x_new_poly) 

plt.plot(x1,y1,'b')
plt.plot(x_new, y_new, "r-", linewidth = 2, label ="Predictions") 
plt.xlabel("$x_1$", fontsize = 18) 
plt.ylabel("$y$", rotation = 0, fontsize = 18) 
plt.legend(loc ="upper left", fontsize = 14) 

plt.title("Biquadratic_predictions_plot") 
plt.show() 


# model evaluation 

mse_deg2 = mean_squared_error(y_new, y1) 
r2_deg2 = r2_score(y_new, y1)   

print('MSE of Polynomial Regression model', mse_deg2)   
print('R2 score of Polynomial regression model: ', r2_deg2) 
 