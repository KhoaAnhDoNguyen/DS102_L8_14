import streamlit as st
import joblib
from PIL import Image
import numpy as np

st.title('Diamond Price Prediction')

image = Image.open('diamond-price-prediction.png')
st.image(image)

model = joblib.load('model_dt.joblib')

st.header('Input Diamond Information')
carat = st.number_input('Insert Carat')

depth = st.number_input('Insert Depth')
table = st.number_input('Insert Table')
x = st.number_input('Insert X')
y = st.number_input('Insert Y')
z = st.number_input('Insert Z')

if carat is not None and depth is not None and table is not None and x is not None and y is not None and z is not None:
    if st.button('Predict'):
        feature_vector = np.array([carat, depth, table, x, y, z]).reshape(1, -1)
        result = str((model.predict(feature_vector)[0])[0])

        st.header('Result')
        st.text(result)
