import streamlit as st
from PIL import Image
import os

def show():
    # --- 1. ENCABEZADO CON ESTILO ---
    st.markdown("""
        <div style='text-align: center;'>
            <h1>🍇 Viña CIA: Certificación Inteligente</h1>
            <p style='font-size: 1.2em; color: gray;'>
                Revolucionando la tenología con Inteligencia Artificial
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    # --- 2. EL CONTEXTO (PROBLEMA VS SOLUCIÓN) ---
    # Usamos columnas para que sea más legible y comparativo
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ⚠️ El Desafío Actual")
        st.warning(
            """
            **Cuello de Botella en Certificación:**
            
            Actualmente, la certificación de calidad de las barricas es un proceso 
            manual, lento y subjetivo.
            
            * Depende 100% de expertos humanos.
            * Propenso a fatiga y errores.
            * Limita la velocidad de exportación.
            """
        )

    with col2:
        st.markdown("### 🚀 Nuestra Solución")
        st.success(
            """
            **Sommelier Artificial (Redes Neuronales):**
            
            Implementamos un modelo de Deep Learning capaz de analizar 
            parámetros físico-químicos en milisegundos.
            
            * **Objetivo:** Clasificación automática (Premium vs Estándar).
            * **Tecnología:** Perceptrón Multicapa (MLP) con PyTorch.
            * **Resultado:** Certificación instantánea y objetiva.
            """
        )

    # --- 3. IMAGEN ILUSTRATIVA (BANNER) ---
    st.markdown("---")
    
    # Intentamos cargar la imagen local 'barricas.jpg' que tenías en tu código original.
    # Si no existe, usamos una de internet para que la página no se rompa.
    try:
        # Ajusta la ruta si tu imagen está dentro de ui/ o data/
        if os.path.exists("barricas.jpg"):
            st.image("barricas.jpg", use_container_width=True, caption="Bodega de Barricas - Proceso de Maduración")
        elif os.path.exists("ui/barricas.jpg"):
             st.image("ui/barricas.jpg", use_container_width=True, caption="Bodega de Barricas")
        else:
            # Imagen de fallback profesional
            st.image("https://images.unsplash.com/photo-1506377247377-2a5b3b417ebb?q=80&w=2070&auto=format&fit=crop", 
                     use_container_width=True, 
                     caption="Bodega Inteligente | Fuente: Unsplash")
    except:
        st.error("No se pudo cargar la imagen de referencia.")

    # --- 4. METODOLOGÍA VISUAL (PASO A PASO) ---
    st.markdown("### ⚙️ Flujo de Trabajo del Proyecto")
    
    paso1, paso2, paso3, paso4 = st.columns(4)
    
    with paso1:
        st.markdown("#### 1. Datos 📊")
        st.caption("Extracción de datos físico-químicos (Acidez, pH, Alcohol) desde laboratorio.")
        
    with paso2:
        st.markdown("#### 2. Limpieza 🧹")
        st.caption("Preprocesamiento y normalización de datos para eliminar ruido.")
        
    with paso3:
        st.markdown("#### 3. IA 🧠")
        st.caption("Entrenamiento de Red Neuronal (MLP) para detectar patrones complejos.")
        
    with paso4:
        st.markdown("#### 4. App 📱")
        st.caption("Despliegue en interfaz web para uso directo de los enólogos.")

    st.markdown("---")

    # --- 5. OBJETIVOS DE NEGOCIO (KPIs) ---
    # Esto simula métricas de éxito del proyecto
    st.subheader("🎯 Impacto Esperado")
    kpi1, kpi2, kpi3 = st.columns(3)
    
    kpi1.metric(label="Precisión del Modelo", value="> 95%", delta="Objetivo")
    kpi2.metric(label="Tiempo de Análisis", value="< 1 seg", delta="-99% vs Humano")
    kpi3.metric(label="Automatización", value="100%", delta="Full AI")

    st.markdown("---")
    
    # --- 6. FOOTER / EQUIPO ---
    st.markdown("""
        <div style='text-align: center; color: grey; font-size: 0.8em;'>
            Desarrollado por <b>Cristian, Ignacio y Antonia </b> | Universidad Tecnologica Metropolitana <br>
            Proyecto de Evaluación 2 
        </div>
    """, unsafe_allow_html=True)