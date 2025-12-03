import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    # URL directa al archivo CSV del repositorio UCI
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=';')
    return df

def show():
    st.title("Análisis Exploratorio de Datos (EDA)")
    st.caption("Explora el dataset de vinos y descubre patrones clave en sus variables químicas.")
    st.markdown("---")

    # Carga de datos
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return

    # Sección 1: Exploración Inicial
    st.header("1. Exploración Inicial del Dataset")
    
    with st.expander("Ver primeras filas del dataset (Raw Data)"):
        st.dataframe(df.head())
    
    st.subheader("Estadísticas Descriptivas")
    st.dataframe(df.describe())

    st.markdown("---")

    # Sección 2: Visualización de Distribuciones
    st.header("2. Distribución de la Calidad (Variable Objetivo)")
    
    col_dist1, col_dist2 = st.columns([2, 1])
    
    with col_dist1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='quality', data=df, palette='Reds', ax=ax)
        ax.set_title("Frecuencia de Calidad de Vinos (0-10)", fontsize=14)
        ax.set_xlabel("Calidad", fontsize=12)
        ax.set_ylabel("Cantidad de Vinos", fontsize=12)
        # Quitar bordes para estilo más limpio
        sns.despine()
        st.pyplot(fig)
    
    with col_dist2:
        st.markdown("### Desbalance de Clases")
        st.info(
            """
            **Observación Crítica:**
            
            El gráfico muestra un claro desbalance. La gran mayoría de los vinos son de calidad "media" (5 y 6).
            
            Hay muy pocos ejemplos de vinos excelentes (8) o muy malos (3), lo que dificulta que el modelo aprenda estas categorías extremas sin procesamiento previo.
            """
        )

    st.markdown("---")

    # Sección 3: Transformación de Datos (DTS)
    st.header("3. Transformación de Datos (DTS) para PyTorch")
    
    # Creación de la columna target
    df['target'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
    
    col_dts1, col_dts2 = st.columns(2)
    
    with col_dts1:
        st.markdown("### Estrategia de Binarización")
        st.warning(
            """
            **Necesidad Técnica:**
            Para entrenar nuestro **Perceptrón Multicapa**, hemos transformado el problema de regresión/multiclase a uno binario:
            
            - **1 (Premium):** Calidad $\ge$ 7
            - **0 (Estándar):** Calidad < 7
            
            Esta simplificación permite que la Red Neuronal converja más rápido y tenga una métrica de éxito más clara para el negocio.
            """
        )
        
    with col_dts2:
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        counts = df['target'].value_counts()
        labels = ['Estándar (0)', 'Premium (1)']
        # Colores corporativos: Burdeo para estándar, Dorado para premium
        colors = ['#800020', '#D4AF37'] 
        
        ax2.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=(0, 0.1), shadow=True)
        ax2.set_title("Proporción Vinos Premium vs Estándar")
        st.pyplot(fig2)

    st.markdown("---")

    # Sección 4: Análisis de Correlación
    st.header("4. Análisis de Correlación")
    st.write("Mapa de calor para identificar qué variables químicas influyen más en la calidad.")
    
    # Calcular correlación
    corr = df.corr()
    
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    # Mascara para el triángulo superior para limpiar el gráfico
    mask = corr.isnull()
    sns.heatmap(corr, annot=True, cmap='RdBu_r', fmt=".2f", ax=ax3, linewidths=0.5)
    st.pyplot(fig3)
    
    # Identificar top 3 correlaciones con 'quality' (excluyendo quality y target)
    correlations = corr['quality'].drop(['quality', 'target'])
    # Ordenar por valor absoluto para ver fuerza de correlación (positiva o negativa)
    top_features = correlations.abs().sort_values(ascending=False).head(3)
    top_names = top_features.index.tolist()
    
    st.markdown("### Variables Químicas Clave")
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    
    # Mostrar métricas de las top 3
    for i, col in enumerate([col_kpi1, col_kpi2, col_kpi3]):
        var_name = top_names[i]
        corr_val = correlations[var_name]
        with col:
            st.metric(label=f"Top {i+1}: {var_name}", value=f"{corr_val:.3f} Corr")
            
    st.success(
        f"""
        **Insight para el Sommelier Artificial:**

        Las variables **{top_names[0]}**, **{top_names[1]}** y **{top_names[2]}** son las que tienen mayor impacto en la calidad.

        *Sugerencia:* Al usar el módulo de predicción, preste especial atención a estos parámetros químicos.
        """
    )
