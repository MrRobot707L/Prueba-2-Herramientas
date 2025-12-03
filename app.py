import streamlit as st
from ui import pantalla_principal, analisis_datos, prediccion, analitica
from datetime import datetime
import json
import os

st.set_page_config(
    page_title="Viña CIA",
    page_icon="",
    layout="wide"
)

# --- Archivo para persistir datos de analitica ---
ANALYTICS_FILE = "data/analytics_history.json"

def cargar_historial_sesiones():
    """Carga el historial de sesiones desde archivo JSON"""
    if os.path.exists(ANALYTICS_FILE):
        try:
            with open(ANALYTICS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def guardar_historial_sesiones(historial):
    """Guarda el historial de sesiones en archivo JSON"""
    os.makedirs(os.path.dirname(ANALYTICS_FILE), exist_ok=True)
    with open(ANALYTICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(historial, f, ensure_ascii=False, indent=2)

# --- ANALITICA WEB: Tracking de sesiones y navegacion ---
if 'session_start' not in st.session_state:
    st.session_state.session_start = datetime.now()
    st.session_state.page_views = {'Inicio': 0, 'Análisis de Datos': 0, 'Predicción IA': 0, 'Analitica': 0}
    st.session_state.total_views = 0
    st.session_state.historial_predicciones = []
    st.session_state.historial_sesiones = cargar_historial_sesiones()

# Sidebar navigation
st.sidebar.title("Navegación")
opcion = st.sidebar.radio(
    "Ir a:",
    ["Inicio", "Análisis de Datos", "Predicción IA", "Analitica"]
)

# Registrar visita a la pagina
st.session_state.page_views[opcion] += 1
st.session_state.total_views += 1

# --- Guardar sesion automaticamente ---
def guardar_sesion_actual():
    """Guarda la sesion actual automaticamente"""
    tiempo_sesion = datetime.now() - st.session_state.session_start
    h = int(tiempo_sesion.total_seconds() // 3600)
    m = int((tiempo_sesion.total_seconds() % 3600) // 60)
    s = int(tiempo_sesion.total_seconds() % 60)
    
    sesion_actual = {
        'inicio': st.session_state.session_start.strftime("%Y-%m-%d %H:%M:%S"),
        'duracion': f"{h:02d}:{m:02d}:{s:02d}",
        'navegaciones': st.session_state.total_views,
        'predicciones': len(st.session_state.historial_predicciones),
        'premium_encontrados': sum(1 for p in st.session_state.historial_predicciones if p.get('clasificacion') == 'Premium')
    }
    
    # Buscar si ya existe esta sesion (por fecha de inicio) y actualizarla
    sesion_inicio = sesion_actual['inicio']
    historial = st.session_state.historial_sesiones
    
    # Actualizar sesion existente o agregar nueva
    encontrada = False
    for i, ses in enumerate(historial):
        if ses['inicio'] == sesion_inicio:
            historial[i] = sesion_actual
            encontrada = True
            break
    
    if not encontrada:
        historial.append(sesion_actual)
    
    # Guardar en archivo
    guardar_historial_sesiones(historial)

# Guardar sesion cada navegacion
guardar_sesion_actual()

# Navigation logic
if opcion == "Inicio":
    pantalla_principal.show()
elif opcion == "Análisis de Datos":
    analisis_datos.show()
elif opcion == "Predicción IA":
    prediccion.show()
elif opcion == "Analitica":
    analitica.show()

# --- ANALITICA WEB: Panel de estadisticas en sidebar ---
st.sidebar.markdown("---")
st.sidebar.markdown("### Analitica de Sesion")

# Reloj en tiempo real en sidebar usando fragment con context manager
@st.fragment(run_every=1)
def reloj_sidebar():
    tiempo_sesion = datetime.now() - st.session_state.session_start
    horas = int(tiempo_sesion.total_seconds() // 3600)
    minutos = int((tiempo_sesion.total_seconds() % 3600) // 60)
    segundos = int(tiempo_sesion.total_seconds() % 60)
    st.metric("⏱️ Tiempo de sesion", f"{horas:02d}:{minutos:02d}:{segundos:02d}")

with st.sidebar:
    reloj_sidebar()
    st.metric("🔄 Total navegaciones", st.session_state.total_views)

# Desglose por pagina
st.sidebar.markdown("**Visitas por seccion:**")
for pagina, visitas in st.session_state.page_views.items():
    st.sidebar.text(f"  {pagina}: {visitas}")

# Nota sobre integracion cloud
st.sidebar.markdown("---")
st.sidebar.caption("Datos cargados desde UCI ML Repository (cloud)")