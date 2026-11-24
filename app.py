import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# Configuración de la página
st.set_page_config(page_title="Dashboard Hospital Lima Norte", layout="wide")

# Cargar datos
df = pd.read_csv('citas_medicas.csv')
df['fecha'] = pd.to_datetime(df['fecha'])
df['mes'] = df['fecha'].dt.month

st.title('🏥 Dashboard de Gestión de Citas Médicas - Hospital Lima Norte')

# Métricas principales (KPIs)
col1, col2, col3 = st.columns(3)
total_citas = len(df)
tasa_cancelacion = (df['estado'] == 'Cancelada').mean() * 100
tasa_ausencia = (df['estado'] == 'No asistió').mean() * 100

col1.metric("Total de Citas Registradas", f"{total_citas:,}")
col2.metric("Tasa de Cancelación", f"{tasa_cancelacion:.1f}%")
col3.metric("Tasa de Ausentismo", f"{tasa_ausencia:.1f}%")

st.divider()

# Gráfico 1: Citas por Especialidad
st.subheader('Citas por Especialidad')
df_esp = df['especialidad'].value_counts().reset_index()
df_esp.columns = ['Especialidad', 'Cantidad']

fig1 = px.bar(
    df_esp,
    x='Especialidad',
    y='Cantidad',
    color='Especialidad',
    text_auto=True
)
st.plotly_chart(fig1, use_container_width=True)

# Gráfico 2: Estado de las Citas
st.subheader('Estado de las Citas')
fig2 = px.pie(
    df,
    names='estado',
    hole=0.3
)
st.plotly_chart(fig2, use_container_width=True)

# Gráfico 3: Citas por Mes
st.subheader('Evolución de Citas por Mes')
citas_mes = df.groupby('mes').size().reset_index(name='cantidad')
fig3 = px.line(
    citas_mes,
    x='mes',
    y='cantidad',
    markers=True
)
st.plotly_chart(fig3, use_container_width=True)

# Sección Predictiva
st.subheader('Proyección de Demanda (Regresión Lineal)')
x = np.array(citas_mes['mes']).reshape(-1, 1)
y = np.array(citas_mes['cantidad'])
modelo = LinearRegression().fit(x, y)
prediccion = modelo.predict([[13]])[0]

st.info(f"📈 **Predicción para el próximo mes (Mes 13):** Se estiman aproximadamente **{int(prediccion)} citas médicas**.")