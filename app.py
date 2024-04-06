
import numpy as np
import pickle
import pandas as pd
import streamlit as st 
from PIL import Image


pickle_in = open("model.pkl","rb")
classifier=pickle.load(pickle_in)


def welcome():
    return "Welcome All"


def predict_note_authentication(age,sex,cigperday,stroke,diab,Chol,BMI,heartrate,glucose,pulsepress):
    
    prediction=classifier.predict([[age,sex,cigperday,stroke,diab,Chol,BMI,heartrate,glucose,pulsepress]])
    print(prediction)
    return prediction

def main():
    st.title("Cardiovascular Disease Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Will you have cardiovascular diseases? </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
   
    age = st.slider("Age", 32, 70,32)

    gender = st.selectbox("Gender",options=['Male' , 'Female'])
  
    cigperday = st.slider("Cigarets Per Day", 0, 70,1)
    prevstro = st.selectbox("Prevelant Stroke",options=['Yes' , 'No'])
    
    diabetes = st.selectbox("Diabetes",options=['Yes' , 'No'])
    Chol = st.slider("Cholestrol Value",113,600,110)
    
    BMI= st.slider("BMI Value",15.96,57.86,15.00)
    
    heartrate= st.slider("Heart Rate Value",45,113,45)
    
    glucose = st.slider("Glucose Value",40,395,40)
    
    pulsepress = st.slider("Pulse Pressure Value",15,160,15)
    
    
    sex = 1
    if gender == 'Male':
        sex = 1
    else:
        sex = 0


    stroke = 1
    if prevstro == 'Yes':
        stroke = 1
    else:
        stroke = 0
    
    diab = 1
    if diabetes == 'Yes':
        diab = 1
    else:
        diab = 0
        
    
    

        
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(age,sex,cigperday,stroke,diab,Chol,BMI,heartrate,glucose,pulsepress)
        if result == 0:
            st.subheader('You will not have a risk of Cardiovascular Disease')
        else:
            st.subheader('It is probable that you will have Cardiovascular Disease in future.')   
    
    if st.button("About"):
        st.text("https://github.com/DokiVamsiKrishna")
        st.text("")

if __name__=='__main__':
    main()