import streamlit as st
import pickle as pkl
import numpy as np

st.title('Diamond Price Prediction')

image = Image.open('diamond-price-prediction.png')
st.image(image)

input = open('model_dt.pkl', 'rb')
model = pkl.load(input)

st.header('Input Diamond Information')
carat = st.number_input('Insert Carat')

options = ['Ideal', 'Premium', 'Good', 'Very Good', 'Fair']
cut = st.selectbox("Select Cut's option", options)
labels_options = {'Ideal' : 0, 'Premium' : 1, 'Good' : 2, 'Very Good' : 3, 'Fair' : 4}
cut.map(labels_options)

options = ['E', 'I', 'J', 'H', 'F', 'G', 'D']
color = st.selectbox("Select Color's option", options)
labels_options = {'E':0, 'I':1, 'J':2, 'H':3, 'F':4, 'G':5, 'D':6}
color.map(labels_options)

options = ['SI2', 'SI1', 'VS1', 'VS2', 'VVS2', 'VVS1', 'I1', 'IF']
clarity = st.selectbox("Select Clarity's option", options)
labels_options = {'SI2':0, 'SI1':1, 'VS1':2, 'VS2':3, 'VVS2':4, 'VVS1':5, 'I1':6, 'IF':7}
clarity.map(labels_options)

depth = st.number_input('Insert Depth')
table = st.number_input('Insert Table')
x = st.number_input('Insert X')
y = st.number_input('Insert Y')
z = st.number_input('Insert Z')

if carat is not None and cut is not None and color is not None and clarity is not None and depth is not None and table is not None and x is not None and y is not None and z is not None:
    if st.button('Predict'):
        feature_vector = np.array([carat, cut, color, clarity, depth, table, x, y, z]).reshape(1,-1)
        result = str((model.predict(feature_vector)[0])[0])

        st.header('Result')
        st.text(result)
