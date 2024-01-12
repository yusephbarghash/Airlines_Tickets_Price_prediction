
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from category_encoders import BinaryEncoder
from xgboost import XGBRegressor
import joblib

columns = joblib.load('inputs.pkl')
model = joblib.load('model.pkl')

def prediction(Airline, Source, Destination, Duration, Total_Stops, Additional_Info, Month, Day):
    df = pd.DataFrame(columns= columns)
    
    df.at[0,'Airline'] = Airline
    df.at[0,'Source'] = Source
    df.at[0,'Destination'] = Destination
    df.at[0,'Duration'] = Duration
    df.at[0,'Total_Stops'] = Total_Stops
    df.at[0,'Additional_Info'] = Additional_Info
    df.at[0,'Month'] = Month
    df.at[0,'Day'] = Day
    
    prediction = model.predict(df)
    return int(prediction[0])

def main():
    st.title('Indian Air Flight Price Prediction')
    
    Airline = st.selectbox('Airline', ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet',
       'Multiple carriers', 'GoAir', 'Vistara', 'Air Asia'])
    
    Source = st.selectbox('Source', ['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai'])
    
    Destination = st.selectbox('Destination', ['New Delhi', 'Banglore', 'Cochin', 'Kolkata', 'Delhi', 'Hyderabad'])
    
    Duration = st.slider('Duration', min_value=70, max_value=3000, step = 1, value = 360)
    
    Total_Stops = st.slider('Total_Stops', min_value=0, max_value=5, step = 1, value = 0)
    
    Additional_Info = st.selectbox('Additional_Info', ['no info', 'in-flight meal not included',
       'no check-in baggage included', 'business'])
    
    Month = st.selectbox('Month', ['March', 'April', 'May', 'June'])
    
    Day = st.selectbox('Day', ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
    
    if st.button("Predict"):
        result = prediction(Airline, Source, Destination, Duration, Total_Stops, Additional_Info, Month, Day)
        st.text(f"The expected Price for this Flight is  : {int(result) + 1}")
    
main()
    
