import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import os

# Configuración de la página
st.set_page_config(page_title="AI Job Market Analysis", layout="wide", page_icon="📊")

# --- FUNCIÓN DE CARGA DE DATOS OPTIMIZADA ---
@st.cache_data
def load_data():
    # Descarga el dataset (esto evita el error 404 de versiones específicas)
    path = kagglehub.dataset_download("shree0910/ai-and-data-science-job-market-dataset-20202026")
    
    # Busca el archivo CSV en la carpeta descargada
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if not files:
        st.error("No se encontró el archivo CSV en el dataset.")
        return pd.DataFrame()
    
    full_path = os.path.join(path, files[0])
    df = pd.read_csv(full_path)
    
    # LIMPIEZA DE COLUMNAS: 
    # Pasamos todo a minúsculas y quitamos espacios para evitar KeyErrors
    df.columns = df.columns.str.strip().str.lower()

    # Mapeo de nombres para que coincidan con el código del dashboard
    # Ajustamos según los nombres comunes en este dataset (year, job_title, salary_in_usd)
    column_mapping = {
        'year': 'Year',
        'job_title': 'Job_Title',
        'salary_in_usd': 'Salary_USD',
        'salary': 'Salary_USD', # Por si acaso se llama solo salary
        'experience_level': 'Experience'
    }
    df = df.rename(columns=column_mapping)
    
    # Aseguramos que 'Year' sea numérico
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        
    return df

# Ejecutar carga
try:
    df = load_data()
except Exception as e:
    st.error(f"Error crítico al conectar con Kaggle: {e}")
    df = pd.DataFrame()

# --- ESTILOS CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

# --- NAVEGACIÓN ---
st.sidebar.title("📊 Navegación")
menu = st.sidebar.radio("Ir a:", ["🏠 Inicio", "📈 Panel de Análisis", "📑 Documentación"])

# --- SECCIÓN 1: LANDING PAGE ---
if menu == "🏠 Inicio":
    st.title("🚀 Análisis del Mercado Laboral en IA & Data Science")
    st.markdown("### Tendencias Estratégicas 2020 - 2026")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.write("")
        st.markdown(f"""
        Bienvenido al portal de inteligencia de datos. Este panel interactivo analiza la evolución 
        de las oportunidades laborales en el ecosistema de la Inteligencia Artificial.
        
        **Lo que encontrarás aquí:**
        * 💰 **Análisis Salarial:** Comparativa por roles y años.
        * 📈 **Tendencias de Mercado:** Crecimiento de la demanda.
        * 🔍 **Filtros Dinámicos:** Explora datos específicos.
        
        **Realizado por:** **Maria Angela Arrieta A.** *Talento Tech - Nivel Integrador*
        """)
        if st.button("🚀 Comenzar Análisis"):
            st.balloons()
            
    with col2:
        # Imagen representativa profesional
        st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&q=80&w=1000", 
                 caption="Data Science Transformation", use_container_width=True)

# --- SECCIÓN 2: DASHBOARD ---
elif menu == "📈 Panel de Análisis":
    if df.empty:
        st.warning("No hay datos disponibles para mostrar.")
    else:
        st.header("📈 Dashboard de Insights")
        
        # Sidebar Filtros
        st.sidebar.header("Filtros")
        if 'Year' in df.columns:
            years = st.sidebar.multiselect(
                "Seleccionar Años", 
                options=sorted(df['Year'].dropna().unique()), 
                default=sorted(df['Year'].dropna().unique())
            )
            df_filtered = df[df['Year'].isin(years)]
        else:
            df_filtered = df

        # Métricas principales
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total de Registros", len(df_filtered))
        with c2:
            avg_salary = df_filtered['Salary_USD'].mean() if 'Salary_USD' in df_filtered.columns else 0
            st.metric("Salario Promedio (USD)", f"${avg_salary:,.0f}")
        with c3:
            roles = df_filtered['Job_Title'].nunique() if 'Job_Title' in df_filtered.columns else 0
            st.metric("Diversidad de Roles", roles)

        st.divider()

        # Gráfico 1: Evolución Salarial
        st.subheader("📊 Salario Promedio por Año")
        st.help("Este gráfico muestra cómo han variado las compensaciones económicas en el tiempo.")
        if 'Year' in df_filtered.columns and 'Salary_USD' in df_filtered.columns:
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            sns.lineplot(data=df_filtered, x='Year', y='Salary_USD', marker='s', color='#1f77b4', ax=ax1)
            plt.title("Tendencia Salarial Anual", fontsize=12)
            st.pyplot(fig1)
        else:
            st.info("Las columnas necesarias para este gráfico no están disponibles.")

        # Gráfico 2: Top Roles
        st.subheader("🏆 Los 10 Roles más Comunes")
        st.help("Muestra la frecuencia de aparición de cada título laboral en el dataset.")
        if 'Job_Title' in df_filtered.columns:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            top_roles = df_filtered['Job_Title'].value_counts().head(10)
            sns.barplot(x=top_roles.values, y=top_roles.index, palette="mako", ax=ax2)
            st.pyplot(fig2)

# --- SECCIÓN 3: DOCUMENTACIÓN ---
elif menu == "📑 Documentación":
    st.header("📑 Documentación y Detalles Técnicos")
    
    t1, t2 = st.tabs(["📖 Diccionario de Datos", "🛠️ Metodología"])
    
    with t1:
        st.write("Estructura detectada del dataset:")
        st.write(df.dtypes)
        st.markdown("""
        **Columnas Clave:**
        - **Year:** Año de la recolección o proyección.
        - **Job_Title:** Nombre del cargo profesional.
        - **Salary_USD:** Salario anual convertido a dólares.
        """)
        
    with t2:
        st.markdown("""
        ### Proceso de Análisis
        1. **Ingesta:** Uso de la API de `kagglehub` para datos en tiempo real.
        2. **Limpieza:** Estandarización de nombres de columnas (Lower case & Strip).
        3. **Visualización:** Implementación de `Seaborn` para análisis estadístico visual.
        4. **Entorno:** Streamlit para despliegue en la nube.
        """)

st.sidebar.markdown("---")
st.sidebar.info(f"**Desarrollado por:**\nMaria Angela Arrieta A.")
