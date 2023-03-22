# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:40:54 2023

@author: VINNY
"""

base="dark"
backgroundColor="#291be6"
secondaryBackgroundColor="#959de8"




import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt


def cost_function(theta, X, y):
    # number of samples
    m = len(y)

  # predicted values
    y_hat = X @ theta

  # sum of squared differences
    cost = 1/(2*m) * np.sum((y_hat - y)**2)

    return cost


def gradient_descent(X, y, theta, alpha, num_iters):
    # number of samples
    m = len(y)

    # empty array to store the cost at each iteration
    costs = []

    for i in range(num_iters):
        # predicted values
        y_hat = X @ theta
        
        # calculate the derivative of the cost function with respect to each model parameter
        derivative = X.T @ (y_hat - y) / m
    
        # update the model parameters
        theta = theta - alpha * derivative

        # calculate the cost
        cost = cost_function(theta, X, y)

        # store the cost at each iteration
        costs.append(cost)

    return theta, costs


class MultipleLinearRegression:
    
    def _init_(self):
        self.theta = None

    def fit(self, X, y, alpha, num_iters):
        # add a column of ones to X
        X = np.column_stack((np.ones(len(X)), X))
        
        
        # initialize the model parameters
        theta = np.zeros(X.shape[1])

        # run gradient descent
        theta, costs = gradient_descent(X, y, theta, alpha, num_iters)

        # store the model parameters
        self.theta = theta

    def predict(self, X):
        # add a column of ones to X
        X = np.column_stack((np.ones(len(X)), X))

        # make predictions using the model parameters
        y_hat = X @ self.theta

        return y_hat

# create a MultipleLinearRegression object
model = MultipleLinearRegression()


a = open('trained1_model.pkl', 'rb')
loaded_model = pickle.load(a)

df = pd.read_csv("df.csv")

# create a Streamlit app
st.title('Melbourne Housing Price Predictor')

# display the table and line chart
if st.checkbox("Show Tables"):
    st.table(df)




# create a function to make predictions
def predict_price(loaded_model, rooms, distance, bathroom, car, landsize, buildingarea):
    data = [[rooms, distance, bathroom, car, landsize, buildingarea]]
    data = StandardScaler().fit_transform(data)
    price = loaded_model.predict(data)[0]
    return price


# add input widgets
rooms = st.number_input('Number of rooms')
distance = st.number_input('Distance from CBD (km)')
bathroom = st.number_input('Number of bathrooms')
car = st.number_input('Number of car spaces')
landsize = st.number_input('Landsize (sqm)')
buildingarea = st.number_input('Building area (sqm)')


# make a prediction and display the result
if st.button('Predict'):
    price = predict_price(loaded_model, rooms, distance, bathroom, car, landsize, buildingarea)
    st.success(f'Estimated price:{round(price)} $')


if st.checkbox("Show the graph"):
    st.line_chart(df["Price"])



# create and display the scatter plot
#fig,ax = plt.subplots()
#ax.bar_chart(df['Rooms'],df['Price'])
#plt.title('scatter')
#st.pyplot(fig)