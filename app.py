import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# 1. Configuración general de la página
st.set_page_config(
    page_title="Dashboard Hospital Lima Norte",
    page_icon="🏥",
    layout="wide"
)

# 2. Cargar y preparar datos
df = pd.read_csv('citas_medicas.csv')
df['fecha'] = pd.to_datetime(df['fecha'])
df['mes'] = df['fecha'].dt.month

# --- BARRA LATERAL DE FILTROS ---
st.sidebar.header("🔍 Filtros de Búsqueda")
st.sidebar.markdown("Filtra los datos del Hospital Lima Norte:")

# Filtro 1: Especialidad
especialidades_disp = sorted(df['especialidad'].unique())
especialidad_sel = st.sidebar.multiselect(
    "Especialidad Médica:",
    options=especialidades_disp,
    default=especialidades_disp
)

# Filtro 2: Estado de la cita
estados_disp = sorted(df['estado'].unique())
estado_sel = st.sidebar.multiselect(
    "Estado de la Cita:",
    options=estados_disp,
    default=estados_disp
)

# Filtro 3: Rango de Edad
edad_min, edad_max = int(df['edad'].min()), int(df['edad'].max())
rango_edad = st.sidebar.slider(
    "Rango de Edad del Paciente:",
    min_value=edad_min,
    max_value=edad_max,
    value=(edad_min, edad_max)
)

# Aplicar todos los filtros al DataFrame
df_filtrado = df[
    (df['especialidad'].isin(especialidad_sel)) &
    (df['estado'].isin(estado_sel)) &
    (df['edad'].between(rango_edad[0], rango_edad[1]))
]

# --- CUERPO PRINCIPAL DEL DASHBOARD ---
st.title('🏥 Dashboard de Gestión de Citas Médicas - Hospital Lima Norte')
st.markdown("Panel interactivo de Inteligencia de Negocios para el monitoreo de demanda y ausentismo.")

# Validar que existan datos tras filtrar
if df_filtrado.empty:
    st.warning("⚠️ No hay registros que coincidan con los filtros seleccionados. Selecciona al menos una opción en el panel izquierdo.")
    st.stop()

# 3. Tarjetas de Métricas Principales (KPIs dinámicos según filtro)
col1, col2, col3, col4 = st.columns(4)
total_citas = len(df_filtrado)
tasa_atencion = (df_filtrado['estado'] == 'Atendida').mean() * 100
tasa_cancelacion = (df_filtrado['estado'] == 'Cancelada').mean() * 100
tasa_ausencia = (df_filtrado['estado'] == 'No asistió').mean() * 100

col1.metric("Total Citas Filtradas", f"{total_citas:,}")
col2.metric("Tasa de Atención", f"{tasa_atencion:.1f}%")
col3.metric("Tasa de Cancelación", f"{tasa_cancelacion:.1f}%")
col4.metric("Tasa de Ausentismo", f"{tasa_ausencia:.1f}%")

st.divider()

# 4. Primera fila de gráficos (Dos columnas lado a lado)
izq, der = st.columns(2)

with izq:
    st.subheader('📊 Citas por Especialidad')
    df_esp = df_filtrado['especialidad'].value_counts().reset_index()
    df_esp.columns = ['Especialidad', 'Cantidad']
    fig1 = px.bar(
        df_esp,
        x='Especialidad',
        y='Cantidad',
        color='Especialidad',
        text_auto=True,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

with der:
    st.subheader('🍩 Distribución por Estado de Cita')
    fig2 = px.pie(
        df_filtrado,
        names='estado',
        hole=0.4,
        color='estado',
        color_discrete_map={
            'Atendida': '#2ecc71',
            'Cancelada': '#f39c12',
            'No asistió': '#e74c3c'
        }
    )
    st.plotly_chart(fig2, use_container_width=True)

# 5. Segunda fila: Evolución mensual y Predicción
st.subheader('📈 Evolución Mensual de Citas')
citas_mes = df_filtrado.groupby('mes').size().reset_index(name='cantidad')
fig3 = px.line(
    citas_mes,
    x='mes',
    y='cantidad',
    markers=True
)
fig3.update_traces(line_color='#2980b9', line_width=3)
st.plotly_chart(fig3, use_container_width=True)

# 6. Sección Predictiva (Regresión Lineal dinámica)
st.subheader('🤖 Proyección de Demanda (Regresión Lineal)')
if len(citas_mes) > 1:
    x = np.array(citas_mes['mes']).reshape(-1, 1)
    y = np.array(citas_mes['cantidad'])
    modelo = LinearRegression().fit(x, y)
    prediccion = modelo.predict([[13]])[0]
    st.info(f"Con los filtros actuales, se proyectan aproximadamente **{int(prediccion):,} citas médicas** para el próximo mes (Mes 13).")
else:
    st.warning("Se necesitan datos de al menos 2 meses distintos para calcular la proyección.")
