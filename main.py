import sklearn 
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
from streamlit_option_menu import option_menu

df=pd.read_csv("/Users/ameen/Documents/PROJECTS_NEW/Crop/Data/cleaned_data.csv")
model=joblib.load("/Users/ameen/Documents/PROJECTS_NEW/Crop/py/data cleaning/crop_pred.pkl")
enc_area=LabelEncoder()
enc_item=LabelEncoder()

area_code = enc_area.fit_transform(df['Area'])
item_code= enc_item.fit_transform(df['Item'])

feature_names = ['Area_encoded', 'Item_encoded', 'Year', 'Area_harvested', 'Yield']

st.set_page_config(page_title="Crop Prediction",
                    layout='wide',
                    initial_sidebar_state='expanded'
                )

st.title(":blue[Predicting Crop Production]")
st.markdown("Using Agricultural Data")

with st.sidebar:
    choice =option_menu("Main Menu",
                        ["Crop Prediction","Visualization"],
                        menu_icon="menu-up",
                       orientation="vertical")
if choice=='Crop Prediction':
    st.header(":blue[Crop Prediction]")
    st.markdown("Choose the input parameters:-")    

    selected_area = st.selectbox('Select Area', enc_area.classes_)
    selected_item = st.selectbox('Select Crop', enc_item.classes_)
    selected_year = st.slider('Select Year', 2000, 2030, 2024)
                   
    area_harvested = st.number_input('Area Harvested in ha',min_value=0.0)
    yield_value = st.number_input('Yield in Kg/ha',min_value=0.0)

    if st.button('Predict'):
                        input_data = pd.DataFrame([[
                                    enc_area.transform([selected_area])[0],
                                    enc_item.transform([selected_item])[0],
                                    selected_year,
                                    area_harvested,
                                    yield_value]], columns=feature_names)
                        prediction=model.predict(input_data)

                        st.header('Predicted Production')
                        st.metric("Production", f"{prediction[0]:.1f} tonnes")

if choice=="Visualization":
                        fig = px.scatter(df, x='Area_harvested', y='Production',color='Item', title='Production Vs Area Harvested')
                        st.plotly_chart(fig)
                        
                        fig = px.area(df, x='Item', y='Production',title='Crop Production Distribution')
                        st.plotly_chart(fig)
                            

