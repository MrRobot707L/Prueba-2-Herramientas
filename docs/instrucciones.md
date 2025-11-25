# Instrucciones para Ejecutar y Probar la Aplicación

Este documento entrega instrucciones detalladas para ejecutar localmente la aplicación desarrollada en Streamlit, así como información sobre su estructura y dependencias.

---

## 1. Requisitos Previos

- Python 3.9 o superior
- pip actualizado
- Entorno virtual recomendado (opcional)

---

## 2. Instalación de Dependencias

Ejecutar en la terminal: pip install -r requirements.txt

## 3. Estructura del Proyecto
app.py
requirements.txt
README.md
ui/
├── init.py
├── pantalla_principal.py
├── analisis_datos.py
└── prediccion.py
data/
└── winequality-red.csv
docs/
├── narrativa.md
├── responsabilidades.md
├── prompts_ia.md
└── instrucciones.md

## 4. Ejecutar la Aplicación

Dentro de la carpeta raíz del proyecto correr: streamlit run app.py

## 5. Navegación

La aplicación contiene tres pantallas principales:

- **Pantalla Principal**: presentación del proyecto.
- **Exploración de Datos**: visualización de tabla, estadísticas y gráficos.
- **Predicción**: modelo simple de ML con entrada del usuario y métricas.

La navegación puede hacerse mediante sidebar o selección de menú, dependiendo de la implementación final.

---

## 6. Notas sobre el Dataset

- El dataset utilizado proviene del UCI Machine Learning Repository.
- Posee 4.898 muestras y 11 características físico-químicas.
- La variable objetivo es la calidad del vino (0–10).
- Puede tratarse como problema de regresión o clasificación.

---

## 7. Frameworks Utilizados

- **Streamlit** para el frontend.
- **Pandas** para manipulación de datos.
- **Matplotlib** para gráficos.
- **Scikit-learn** para el modelo de ML.
- **GitHub** para control de versiones y entrega del proyecto.
