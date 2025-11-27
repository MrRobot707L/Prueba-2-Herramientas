import streamlit as st
from ui import pantalla_principal, analisis_datos, prediccion

st.set_page_config(
    page_title="Viña CIA",
    page_icon="",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("Navegación")
opcion = st.sidebar.radio(
    "Ir a:",
    ["Inicio", "Análisis de Datos", "Predicción IA"]
)

# Navigation logic
if opcion == "Inicio":
    pantalla_principal.show()
elif opcion == "Análisis de Datos":
    analisis_datos.show()
elif opcion == "Predicción IA":
    prediccion.show()