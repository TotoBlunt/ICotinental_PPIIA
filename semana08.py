import streamlit as st 
import pandas as pd 

#Titulo para el app
st.title("K-Means Clustering con Streamlit")

#Subir archivo de excel
upload_file = st.file_uploader('Sube un archivo Excel',type=['xlsx'])

if upload_file is not None:
    #Leer archivo excel
    df = pd.read_excel(upload_file)

    st.write('### Vista previa de los datos')
    st.write(df.head())