import numpy as np
from sklearn.linear_model import LinearRegression

#Generate sythetic data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

#Create and train the model
Model = LinearRegression()
model.fit(X, y)

#Predict using the trained model 
x_test = np.array([[6], [7], [8]])
y_pred = model.predict(x_test)

print(“Predictions: ”, y_pred)
