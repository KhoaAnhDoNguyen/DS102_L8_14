import streamlit as st
import joblib
from PIL import Image
import numpy as np

st.title('Diamond Price Prediction')

image = Image.open('diamond-price-prediction.png')
st.image(image)

model = joblib.load('model_lr.joblib')

st.header('Input Diamond Information')
carat = st.number_input('Insert Carat')

options = ['Ideal', 'Premium', 'Good', 'Very Good', 'Fair']
cut = st.selectbox("Select Cut's option", options)
labels_options_cut = {'Ideal': 0, 'Premium': 1, 'Good': 2, 'Very Good': 3, 'Fair': 4}
cut_value = labels_options_cut[cut]

options = ['E', 'I', 'J', 'H', 'F', 'G', 'D']
color = st.selectbox("Select Color's option", options)
labels_options_color = {'E': 0, 'I': 1, 'J': 2, 'H': 3, 'F': 4, 'G': 5, 'D': 6}
color_value = labels_options_color[color]

options = ['SI2', 'SI1', 'VS1', 'VS2', 'VVS2', 'VVS1', 'I1', 'IF']
clarity = st.selectbox("Select Clarity's option", options)
labels_options_clarity = {'SI2': 0, 'SI1': 1, 'VS1': 2, 'VS2': 3, 'VVS2': 4, 'VVS1': 5, 'I1': 6, 'IF': 7}
clarity_value = labels_options_clarity[clarity]


depth = st.number_input('Insert Depth')
table = st.number_input('Insert Table')
x = st.number_input('Insert X')
y = st.number_input('Insert Y')
z = st.number_input('Insert Z')

if carat is not None and cut is not None and color is not None and clarity is not None and depth is not None and table is not None and x is not None and y is not None and z is not None:
    if st.button('Predict'):
        feature_vector = np.array([carat, cut_value, color_value, clarity_value, depth, table, x, y, z]).reshape(1,-1)
        result = str((model.predict(feature_vector)))

        st.header('Result')
        st.text(result)
