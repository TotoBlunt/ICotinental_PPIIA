import streamlit as st 
import pandas as pd 

#Titulo para el app
st.title("K-Means Clustering con Streamlit")

#Subir archivo de excel
upload_file = st.file_uploader('Sube un archivo Excel',type=['xlsx'])