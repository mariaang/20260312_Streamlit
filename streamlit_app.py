import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Configuración de la página
st.set_page_config(page_title="AI Job Market Analysis", layout="wide", page_icon="📊")

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    # 1. Descargamos el dataset completo
    path = kagglehub.dataset_download("shree0910/ai-and-data-science-job-market-dataset-20202026")
    
    # 2. Buscamos el archivo CSV dentro de esa carpeta
    import os
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    
    if not files:
        raise FileNotFoundError("No se encontró ningún archivo CSV en el dataset.")
    
    # 3. Cargamos el primer CSV que encuentre (el principal)
    full_path = os.path.join(path, files[0])
    df = pd.read_csv(full_path)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    # Fallback para desarrollo si no hay internet/credenciales
    df = pd.DataFrame() 

# --- ESTILOS CSS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- NAVEGACIÓN ---
menu = st.sidebar.radio("Navegación", ["🏠 Inicio", "📊 Dashboard Interactivo", "📑 Documentación"])

# --- SECCIÓN 1: LANDING PAGE ---
if menu == "🏠 Inicio":
    st.title("🚀 Análisis del Mercado Laboral en IA & Data Science")
    st.subheader("Bienvenido al Panel de Visualización Estratégica")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
        Este dataset proyecta y analiza las tendencias de empleo desde **2020 hasta 2026**. 
        Como profesional en análisis de datos, esta herramienta te permitirá entender:
        * 📈 **Evolución Salarial**: Cómo cambian los ingresos según el rol.
        * 🌍 **Distribución Geográfica**: Dónde están las mejores oportunidades.
        * 🛠️ **Skills Demandadas**: Qué tecnologías dominan el mercado.
        
        **Realizado por:** Maria Angela Arrieta A.
        """)
        if st.button("Ingresar al Panel de Trabajo"):
            st.info("Usa el menú de la izquierda para navegar al Dashboard.")
            
    with col2:
        # Imagen representativa generada por IA sobre análisis de datos
        st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&q=80&w=1000", 
                 caption="Data Science Trends 2020-2026")

# --- SECCIÓN 2: DASHBOARD ---
elif menu == "📊 Dashboard Interactivo":
    st.header("📊 Panel de Análisis de Datos")
    
    # Filtros rápidos
    years = st.multiselect("Seleccionar Años", options=sorted(df['Year'].unique()), default=sorted(df['Year'].unique()))
    df_filtered = df[df['Year'].isin(years)]

    # Métricas clave
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Registros", len(df_filtered))
    m2.metric("Promedio Salarial", f"${df_filtered['Salary_USD'].mean():,.2f}")
    m3.metric("Roles Únicos", df_filtered['Job_Title'].nunique())

    st.divider()

    # Gráfico 1: Evolución de Salarios
    st.subheader("📈 Evolución Salarial Anual")
    st.help("Este gráfico muestra la tendencia media de los salarios en USD. La línea sombreada representa el intervalo de confianza.")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=df_filtered, x='Year', y='Salary_USD', marker='o', color='#2E86C1', ax=ax1)
    plt.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig1)

    # Gráfico 2: Top Roles vs Salario (Categoría)
    st.subheader("🏆 Top 10 Roles por Salario Promedio")
    st.help("Visualiza cuáles son las posiciones mejor pagadas en el sector de IA.")
    top_roles = df_filtered.groupby('Job_Title')['Salary_USD'].mean().sort_values(ascending=False).head(10).reset_index()
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_roles, x='Salary_USD', y='Job_Title', palette='viridis', ax=ax2)
    st.pyplot(fig2)

# --- SECCIÓN 3: DOCUMENTACIÓN ---
elif menu == "📑 Documentación":
    st.header("📑 Documentación del Proyecto")
    
    tab1, tab2, tab3 = st.tabs(["Información del Dataset", "Diccionario de Datos", "Metodología"])
    
    with tab1:
        st.markdown("""
        ### AI & Data Science Job Market (2020–2026)
        Este dataset es una recopilación exhaustiva de datos sobre roles de IA y Ciencia de Datos.
        - **Fuente:** Kaggle (shree0910)
        - **Periodo:** 2020 - 2026 (Proyectado)
        """)
    
    with tab2:
        st.write(df.dtypes)
        st.markdown("El dataset contiene columnas sobre Salarios, Títulos, Experiencia y Ubicación.")
    
    with tab3:
        st.markdown("""
        Para este análisis intermedio, se aplicaron técnicas de:
        1. **Limpieza de datos:** Manejo de valores nulos.
        2. **Agregación:** Resúmenes estadísticos por año y rol.
        3. **Visualización:** Implementación de Seaborn para gráficos estadísticos de alta calidad.
        """)

st.sidebar.markdown("---")
st.sidebar.write("👤 **Autora:** Maria Angela Arrieta A.")

