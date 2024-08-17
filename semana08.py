import streamlit as st 
import pandas as pd 

#Titulo para el app
st.title("K-Means Clustering con Streamlit")

#Subir archivo de excel
upload_file = st.file_uploader('Sube un archivo Excel',type=['xlsx'])

if upload_file is not None:
    try: 
        #Leer archivo excel
        df = pd.read_excel(upload_file)

        st.write('### Vista previa de los datos')
        st.write(df.head())

        #Seleccionar columnas categoricas
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

        if categorical_columns:
            st.write('### Columnas categoricas indentificadas')
            st.write(categorical_columns)

            #Convertir columnas categoricas a dummies
            df = pd.get_dummies(df,columns=categorical_columns)
            st.write('### Datos despues de la conversion a dummies')
            st.write(df.head())
        else:
            st.write('No se encontraron columnas categoricas en los datos')
    except Exception as e:
        st.error(f'Error al leer archivo excel: {e}')