import streamlit as st 
import pandas as pd 
import openpyxl
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

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

        # Verificar valores faltantes
        if df.isnull().values.any():
            st.write("### Valores faltantes encontrados")
            st.write(df.isnull().sum())
            
            # Manejo de valores faltantes
            df = df.fillna(df.mean())

            st.write("### Datos después de manejar valores faltantes")
            st.write(df.head())

        # Asegurar que todas las columnas sean numéricas
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        #Normalizar los datos
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)

        #Seleccion del numero de clusters
        st.write('### Selecciona el numero de clusters')
        num_clusters = st.slider('Numero de cluster',min_value=2,max_value=10,value=3) # 3 por defecto

        #Aplicando el K-Means
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(df_scaled)

        #Añadir cluster al DF original
        df['Cluster'] = clusters

        st.write('### Datos con el cluster asignado')
        st.write(df.head())

            # Visualización de los clusters utilizando PCA
        pca_df['Cluster'] = clusters
        fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', title='Visualización de Clusters usando PCA')
        st.plotly_chart(fig)

        # Preparar el archivo CSV en memoria
        csv = df.to_csv(index=False)

        # Crear botón de descarga
        st.download_button(
            label="Descargar CSV con Resultados",
            data=csv,
            file_name='resultados_cluster.csv',
            mime='text/csv'
        )


    except Exception as e:
        st.error(f'Error al leer archivo excel: {e}')