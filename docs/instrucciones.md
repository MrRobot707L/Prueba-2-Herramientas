# Instrucciones para Ejecutar y Probar la Aplicación

Este documento explica cómo instalar, ejecutar y navegar por la aplicación **Viña CIA – Sommelier Artificial**, desarrollada en Streamlit.

---

## 1. Requisitos previos

- Python 3.9 o superior
- `pip` actualizado
- (Recomendado) Entorno virtual de Python

---

## 2. Instalación de dependencias

1. Abrir una terminal en la carpeta raíz del proyecto.
2. (Opcional, pero recomendado) Crear y activar un entorno virtual:

	 ```powershell
	 python -m venv .venv
	 .\.venv\Scripts\Activate.ps1
	 ```

3. Instalar las dependencias del proyecto:

	 ```powershell
	 pip install -r requirements.txt
	 ```

---

## 3. Estructura del proyecto (resumen)

```text
Prueba-2-Herramientas/
|-- app.py                  # Punto de entrada de la aplicación Streamlit
|-- requirements.txt        # Dependencias del proyecto
|-- README.md
|-- ui/
|   |-- pantalla_principal.py
|   |-- analisis_datos.py   # Módulo de Análisis de Datos (EDA)
|   |-- prediccion.py       # Módulo de Predicción IA
|   |-- analitica.py        # Módulo de Analítica Web
|-- data/
|   |-- dataset.csv         # Dataset original (Wine Quality Red)
|   |-- procesado.pkl       # Datos preprocesados para el modelo
|   |-- analytics_history.json  # Historial de uso de la app
|-- docs/
		|-- narrativa.md
		|-- responsabilidades.md
		|-- prompts_ia.md
		|-- instrucciones.md
```

---

## 4. Ejecutar la aplicación

1. Desde la carpeta raíz del proyecto, con el entorno virtual activado, ejecutar:

	 ```powershell
	 streamlit run app.py
	 ```

2. Streamlit abrirá la aplicación en el navegador (por defecto en `http://localhost:8501`).

---

## 5. Navegación por la aplicación

La aplicación contiene cuatro pantallas principales, accesibles desde el menú lateral:

- **Pantalla Principal**  
	Presentación del proyecto, contexto del problema y solución propuesta con IA.

- **Análisis de Datos (EDA)**  
	Exploración del dataset de vinos: tabla, estadísticas descriptivas, distribuciones y mapa de correlaciones.

- **Predicción IA**  
	Entrenamiento y uso del modelo MLP:
	- Entrenar el modelo (curva de aprendizaje y métricas: Accuracy, Precision, Recall, F1).
	- Introducir valores de un vino (o usar "Aleatorizar Muestra") para clasificarlo como **Premium** o **Estándar**.
	- Guardar automáticamente cada predicción en un historial.

- **Analítica Web**  
	Panel de métricas de uso de la app:
	- Tiempo de sesión en tiempo real.
	- Navegaciones por sección.
	- Historial de vinos clasificados, gráficos interactivos y exportación a CSV.
	- Resumen de sesiones actuales e históricas.

---

## 6. Notas sobre el dataset

- El dataset proviene del **UCI Machine Learning Repository** (Wine Quality – Red).
- Contiene 1.599 vinos y 11 características físico‑químicas.
- La variable original es la calidad del vino (0–10).
- En esta app se transforma a un problema de **clasificación binaria**:
	- `1` → Vino **Premium** (calidad ≥ 7)
	- `0` → Vino **Estándar** (calidad < 7)

---

## 7. Frameworks y librerías utilizadas

- **Streamlit**: interfaz web y navegación.
- **Pandas**: carga y manipulación de datos.
- **PyTorch**: implementación de la red neuronal MLP.
- **Scikit-learn**: métricas de evaluación del modelo.
- **Matplotlib / Seaborn**: visualizaciones en el EDA.
- **Plotly**: gráficos interactivos en el módulo de analítica.
