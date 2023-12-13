import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import plotly.express as px
import base64

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_background('g2.png')


html_temp = """
<div style="background-color:#5A7D9F;padding:10px">
<h1 style="color:white;text-align:center;">Credit Card Fraud Detection</h1>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)
st.write('\n')

##button_style = """
   # <style>
    #.color-btn {
    #    background-color: #3498db; /* Change this color to your preferred color */
      ##  color: white;
      #  padding: 0.5rem 1rem;
      #  border-radius: 5px;
      ##  border: none;
       # text-align: center;
      #  text-decoration: none;
      #  display: inline-block;
      ###  font-size: 16px;
       # margin: 4px 2px;
       # cursor: pointer;
   # }
   ### </style>
   # <button class="color-btn">Click me!</button>
#  """

# Display the button with custom style
# st.markdown(button_style, unsafe_allow_html=True)


# Use markdown to apply custom CSS to the slider
st.markdown(
    """
    <style>
    /* Change the color of the slider handle */
    .stSlider > div > div > div {
         !important;
    }

    /* Change the color of the slider track */
    .stSlider > div > div > div > div {
        background-color: #5A7D9F !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)






st.sidebar.header('Input Credit Card Details')


   
V1 = st.sidebar.slider('V1', -5.0, 1.5, 5.0)
V2 = st.sidebar.slider('V2', -5.0, 1.5, 5.0)
V3 = st.sidebar.slider('V3', -5.0, 1.5, 5.0)
V4 = st.sidebar.slider('V4', -5.0, 1.5, 5.0)
V5 = st.sidebar.slider('V5', -5.0, 1.5, 5.0)
V6 = st.sidebar.slider('V6', -5.0, 1.5, 5.0)
V7 = st.sidebar.slider('V7', -5.0, 1.5, 5.0)
V8 = st.sidebar.slider('V8', -5.0, 1.5, 5.0)
V9 = st.sidebar.slider('V9', -5.0, 1.5, 5.0)
V10 = st.sidebar.slider('V10', -5.0, 1.5, 5.0)
V11 = st.sidebar.slider('V11', -5.0, 1.5, 5.0)
V12 = st.sidebar.slider('V12', -5.0, 1.5, 5.0)
V13 = st.sidebar.slider('V13', -5.0, 1.5, 5.0)
V14 = st.sidebar.slider('V14', -5.0, 1.5, 5.0)
V15 = st.sidebar.slider('V15', -5.0, 1.5, 5.0)
V16 = st.sidebar.slider('V16', -5.0, 1.5, 5.0)
V17 = st.sidebar.slider('V17', -5.0, 1.5, 5.0)
V18 = st.sidebar.slider('V18', -5.0, 1.5, 5.0)
V19 = st.sidebar.slider('V19', -5.0, 1.5, 5.0)
V20 = st.sidebar.slider('V20', -5.0, 1.5, 5.0)
V21 = st.sidebar.slider('V21', -5.0, 1.5, 5.0)
V22 = st.sidebar.slider('V22', -5.0, 1.5, 5.0)
V23 = st.sidebar.slider('V23', -5.0, 1.5, 5.0)
V24 = st.sidebar.slider('V24', -5.0, 1.5, 5.0)
V25 = st.sidebar.slider('V25', -5.0, 1.5, 5.0)
V26 = st.sidebar.slider('V26', -5.0, 1.5, 5.0)
V27 = st.sidebar.slider('V27', -5.0, 1.5, 5.0)
V28 = st.sidebar.slider('V28', -5.0, 1.5, 5.0)
Time = st.sidebar.slider('Time', 0, 200000,100000)
Amount = st.sidebar.number_input('Amount')


data = {'V1': V1,
        'V2': V2,
        'V3': V3,
        'V4': V4,
        'V5': V5,
        'V6': V6,
        'V7': V7,
        'V8': V8,
        'V9': V9,
        'V10': V10,
        'V11': V11,
        'V12': V12,
        'V13': V13,
        'V14': V14,
        'V15': V15,
        'V16': V16,
        'V17': V17,
        'V18': V18,
        'V19': V19,
        'V20': V20,
        'V21': V21,
        'V22': V22,
        'V23': V23,
        'V24': V24,
        'V25': V25,
        'V26': V26,
        'V27': V27,
        'V28': V28,
        'Amount': Amount
                }
fea = pd.DataFrame(data, index=[0])
        






st.subheader('Credit Card Data')



####
data = {
    'Category': [f"V{i}" for i in range(1, 29)],
    'Inputs': [V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28],
}

df = pd.DataFrame(data)


for column in df.columns[1:]:
    st.write(f"#### {column}")
    fig = px.bar(df, x='Category', y=column, orientation='v', title=f'Bar Chart for {column}', labels={'value': 'Value'})
    fig.update_layout(height=300, width=600)
    st.plotly_chart(fig)

#####
st.write(fea)

load_clf = joblib.load(open('savedModels/model.joblib', 'rb'))

prediction = load_clf.predict(fea)
prediction_probability = load_clf.predict_proba(fea)

st.subheader('Prediction')
if prediction == 0:
    st.write("Normal Transaction")
else:
    st.write("Fraudulent Transaction")

st.subheader('Prediction Probability')
st.write(prediction_probability)

