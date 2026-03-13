import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import os

# Configuración de la página
st.set_page_config(page_title="AI Job Market Analysis", layout="wide", page_icon="📊")

@st.cache_data
def load_data():
    # Descarga directa
    path = kagglehub.dataset_download("shree0910/ai-and-data-science-job-market-dataset-20202026")
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    full_path = os.path.join(path, files[0])
    df = pd.read_csv(full_path)
    
    # Estandarizar columnas a minúsculas y sin espacios
    df.columns = df.columns.str.strip().str.lower()

    # MAPEO INTELIGENTE: Buscamos coincidencias comunes en datasets de salarios de Kaggle
    rename_dict = {}
    
    # Para el Año
    for col in df.columns:
        if col in ['year', 'work_year', 'job_year']:
            rename_dict[col] = 'Year'
        if col in ['salary_in_usd', 'salary_usd', 'salary']:
            rename_dict[col] = 'Salary_USD'
        if col in ['job_title', 'role', 'designation']:
            rename_dict[col] = 'Job_Title'
            
    df = df.rename(columns=rename_dict)
    
    # Asegurar tipos de datos
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    if 'Salary_USD' in df.columns:
        df['Salary_USD'] = pd.to_numeric(df['Salary_USD'], errors='coerce')
        
    return df

# Ejecutar carga
df = load_data()

# --- NAVEGACIÓN ---
st.sidebar.title("📊 Menú")
menu = st.sidebar.radio("Ir a:", ["🏠 Inicio", "📈 Panel de Análisis", "📑 Documentación"])

if menu == "🏠 Inicio":
    st.title("🚀 Análisis del Mercado Laboral en IA")
    st.markdown("### Realizado por: Maria Angela Arrieta A.")
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&q=80&w=1000", use_container_width=True)

elif menu == "📈 Panel de Análisis":
    st.header("📈 Dashboard de Insights")

    # Filtro de Año
    if 'Year' in df.columns:
        years = st.sidebar.multiselect("Filtrar por Año", 
                                      options=sorted(df['Year'].unique()), 
                                      default=sorted(df['Year'].unique()))
        df_filtered = df[df['Year'].isin(years)]
    else:
        df_filtered = df

    # Métricas
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total Registros", len(df_filtered))
    with c2:
        if 'Salary_USD' in df_filtered.columns:
            st.metric("Salario Promedio", f"${df_filtered['Salary_USD'].mean():,.0f}")

    st.divider()

    # --- GRÁFICO 1: EVOLUCIÓN SALARIAL ---
    st.subheader("📊 Salario Promedio por Año")
    # Usamos st.caption en lugar de st.help para evitar el error visual
    st.caption("💡 Este gráfico muestra la tendencia de ingresos anuales en USD.")
    
    if 'Year' in df_filtered.columns and 'Salary_USD' in df_filtered.columns:
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        # Agrupamos para asegurar que el gráfico de líneas salga limpio
        data_plot = df_filtered.groupby('Year')['Salary_USD'].mean().reset_index()
        sns.lineplot(data=data_plot, x='Year', y='Salary_USD', marker='o', color='#0077b6', ax=ax1)
        ax1.set_title("Evolución Salarial (Media)")
        st.pyplot(fig1)
    else:
        st.error("No se encontraron las columnas 'Year' o 'Salary_USD'. Por favor revisa el dataset.")

    # --- GRÁFICO 2: TOP ROLES ---
    st.subheader("🏆 Los 10 Roles más Comunes")
    st.caption("💡 Frecuencia de los cargos más demandados en el mercado.")
    
    if 'Job_Title' in df_filtered.columns:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        top_roles = df_filtered['Job_Title'].value_counts().head(10)
        sns.barplot(x=top_roles.values, y=top_roles.index, palette="viridis", ax=ax2)
        st.pyplot(fig2)
    else:
        st.error("No se encontró la columna 'Job_Title'.")

elif menu == "📑 Documentación":
    st.header("Documentación")
    st.write("Columnas detectadas en el archivo:")
    st.write(list(df.columns))
    st.write(df.head())

st.sidebar.markdown("---")
st.sidebar.write("👤 Maria Angela Arrieta A.")
