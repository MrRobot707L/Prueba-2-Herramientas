import streamlit as st
import pandas as pd
from datetime import datetime
import time
import json
import os

# Archivo para persistir datos
ANALYTICS_FILE = "data/analytics_history.json"

def guardar_historial_sesiones(historial):
    """Guarda el historial de sesiones en archivo JSON"""
    os.makedirs(os.path.dirname(ANALYTICS_FILE), exist_ok=True)
    with open(ANALYTICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(historial, f, ensure_ascii=False, indent=2)

def show():
    st.title("Panel de Analítica Web")
    st.caption("Consulta métricas de uso de la app y el historial de predicciones.")
    st.markdown("---")
    
    # Inicializar historial de predicciones si no existe
    if 'historial_predicciones' not in st.session_state:
        st.session_state.historial_predicciones = []
    
    # Inicializar historial de sesiones si no existe
    if 'historial_sesiones' not in st.session_state:
        st.session_state.historial_sesiones = []
    
    # --- Sección 1: Métricas Generales de Sesión ---
    st.header("1. Métricas de Sesión")
    
    # Calcular valores estaticos para las otras metricas
    premium_count = sum(1 for p in st.session_state.historial_predicciones if p.get('clasificacion') == 'Premium')
    total_predicciones = len(st.session_state.historial_predicciones)
    
    # Layout con reloj en tiempo real usando st.fragment (Streamlit 1.33+)
    @st.fragment(run_every=1)
    def reloj_tiempo_real():
        tiempo_sesion = datetime.now() - st.session_state.session_start
        horas = int(tiempo_sesion.total_seconds() // 3600)
        minutos = int((tiempo_sesion.total_seconds() % 3600) // 60)
        segundos = int(tiempo_sesion.total_seconds() % 60)
        st.metric("⏱️ Tiempo de Sesión", f"{horas:02d}:{minutos:02d}:{segundos:02d}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        reloj_tiempo_real()
    with col2:
        st.metric("🔄 Total Navegaciones", st.session_state.total_views)
    with col3:
        st.metric("🍷 Vinos Analizados", total_predicciones)
    with col4:
        st.metric("🏆 Vinos Premium", premium_count)

    # Historia de usuario resumida
    if total_predicciones > 0:
        st.info(
            f"En esta sesión has clasificado **{total_predicciones}** vinos, "
            f"de los cuales **{premium_count}** fueron Premium."
        )
    
    st.markdown("---")
    
    # --- Sección 2: Navegación por Secciones ---
    st.header("2. Uso por Sección")
    
    col_nav1, col_nav2 = st.columns([1, 2])
    
    with col_nav1:
        st.markdown("**Visitas por pagina:**")
        for pagina, visitas in st.session_state.page_views.items():
            st.metric(pagina, visitas)
    
    with col_nav2:
        # Grafico de barras de navegacion
        if st.session_state.total_views > 0:
            chart_data = pd.DataFrame({
                'Sección': list(st.session_state.page_views.keys()),
                'Visitas': list(st.session_state.page_views.values())
            })
            st.bar_chart(chart_data.set_index('Sección'))
    
    st.markdown("---")
    
    # --- Sección 3: Historial de Predicciones ---
    st.header("3. Historial de Vinos Clasificados")
    
    if len(st.session_state.historial_predicciones) == 0:
        st.info("No se han realizado predicciones aun. Ve a 'Prediccion IA' para clasificar vinos.")
    else:
        # Resumen
        total = len(st.session_state.historial_predicciones)
        premium = sum(1 for p in st.session_state.historial_predicciones if p.get('clasificacion') == 'Premium')
        estandar = total - premium
        
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric("Total Clasificados", total)
        with col_res2:
            st.metric("Premium", premium, delta=f"{(premium/total)*100:.1f}%")
        with col_res3:
            st.metric("Estándar", estandar, delta=f"{(estandar/total)*100:.1f}%")
        
        # Tabla con historial completo
        st.markdown("### Detalle de Predicciones")
        df_historial = pd.DataFrame(st.session_state.historial_predicciones)
        
        # Reordenar columnas para mejor visualizacion
        cols_order = ['timestamp', 'clasificacion', 'probabilidad'] + [c for c in df_historial.columns if c not in ['timestamp', 'clasificacion', 'probabilidad']]
        df_historial = df_historial[cols_order]
        
        # Mostrar tabla
        st.dataframe(df_historial, use_container_width=True)
        
        # Boton para descargar historial
        csv = df_historial.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Historial CSV",
            data=csv,
            file_name="historial_predicciones.csv",
            mime="text/csv"
        )
        
        # Gráfico de distribución de clasificaciones (interactivo con Plotly)
        st.markdown("### Distribución de Clasificaciones")
        import plotly.express as px
        
        df_pie = pd.DataFrame({
            'Clasificación': ['Premium', 'Estándar'],
            'Cantidad': [premium, estandar]
        })
        
        fig = px.pie(
            df_pie, 
            values='Cantidad', 
            names='Clasificación',
            color='Clasificación',
            color_discrete_map={'Premium': '#D4AF37', 'Estándar': '#800020'},
            hole=0.4  # Donut chart
        )
        fig.update_layout(
            height=300,
            width=400,
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=True
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=False)
        
        # Boton para limpiar historial
        if st.button("🗑️ Limpiar Historial de Predicciones"):
            st.session_state.historial_predicciones = []
            st.rerun()
    
    st.markdown("---")
    
    # --- Sección 4: Historial de Sesiones ---
    st.header("4. Historial de Sesiones")
    
    # Calcular duracion actual para el historial
    tiempo_actual = datetime.now() - st.session_state.session_start
    h = int(tiempo_actual.total_seconds() // 3600)
    m = int((tiempo_actual.total_seconds() % 3600) // 60)
    s = int(tiempo_actual.total_seconds() % 60)
    
    # Sesión actual
    sesion_actual = {
        'inicio': st.session_state.session_start.strftime("%Y-%m-%d %H:%M:%S"),
        'duracion': f"{h:02d}:{m:02d}:{s:02d}",
        'navegaciones': st.session_state.total_views,
        'predicciones': len(st.session_state.historial_predicciones),
        'premium_encontrados': premium_count
    }
    
    # Mostrar sesión actual
    st.markdown("### Sesión Actual")
    col_ses1, col_ses2, col_ses3 = st.columns(3)
    with col_ses1:
        st.info(f"**Inicio:** {sesion_actual['inicio']}")
    with col_ses2:
        st.info(f"**Duración:** {sesion_actual['duracion']}")
    with col_ses3:
        st.info(f"**Actividad:** {sesion_actual['navegaciones']} navs, {sesion_actual['predicciones']} preds")
    
    st.caption("💾 La sesión se guarda automáticamente en cada navegación")
    
    # Mostrar historial de sesiones anteriores
    if len(st.session_state.historial_sesiones) > 0:
        st.markdown("### Sesiones Anteriores Guardadas")
        df_sesiones = pd.DataFrame(st.session_state.historial_sesiones)
        st.dataframe(df_sesiones, use_container_width=True)
        
        # Estadísticas acumuladas
        st.markdown("### Estadísticas Acumuladas")
        total_navs = sum(s.get('navegaciones', 0) for s in st.session_state.historial_sesiones)
        total_preds = sum(s.get('predicciones', 0) for s in st.session_state.historial_sesiones)
        total_premium = sum(s.get('premium_encontrados', 0) for s in st.session_state.historial_sesiones)
        
        col_ac1, col_ac2, col_ac3 = st.columns(3)
        with col_ac1:
            st.metric("Total Navegaciones (histórico)", total_navs)
        with col_ac2:
            st.metric("Total Predicciones (histórico)", total_preds)
        with col_ac3:
            st.metric("Total Premium (histórico)", total_premium)
    else:
        st.info("No hay sesiones anteriores guardadas.")
    
    st.markdown("---")
    
    # --- Sección 5: Información del Sistema ---
    st.header("5. Información del Sistema")
    
    col_sys1, col_sys2 = st.columns(2)
    
    with col_sys1:
        st.markdown("**Estado del Modelo:**")
        if st.session_state.get('modelo_entrenado') is not None:
            st.success("Modelo entrenado y listo")
            if 'accuracy' in st.session_state:
                st.write(f"- Accuracy: {st.session_state.accuracy:.2%}")
                st.write(f"- Precision: {st.session_state.precision:.2%}")
                st.write(f"- Recall: {st.session_state.recall:.2%}")
                st.write(f"- F1-Score: {st.session_state.f1:.2%}")
        else:
            st.warning("Modelo no entrenado")
    
    with col_sys2:
        st.markdown("**Fuente de Datos:**")
        st.write("- Dataset: Wine Quality Red (UCI)")
        st.write("- Registros: 1,599 vinos")
        st.write("- Features: 11 variables químicas")
        st.write("- Origen: UCI Machine Learning Repository")
        st.caption("Datos cargados desde fuente externa (cloud)")
