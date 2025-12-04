# Viña CIA - Sommelier Artificial (Streamlit + PyTorch)

**Versión Avanzada (75%)**

Aplicación web en Streamlit para analizar y predecir la calidad de vinos tintos usando un modelo de red neuronal (MLP) y un módulo de analítica de uso en tiempo real.


## Descripción del Proyecto

Este proyecto utiliza el dataset Wine Quality del repositorio UCI Machine Learning para construir un "Sommelier Artificial" que clasifica vinos como Premium o Estándar basándose en sus características físico-químicas.

Breve flujo de uso:
“1. Ve a Análisis de Datos para conocer el dataset.”
“2. Entrena el modelo en Predicción IA.”
“3. Clasifica vinos y revisa estadísticas en Analítica Web.”

## Características clave
Bullets muy breves:
“Clasificación Premium vs Estándar con red neuronal MLP en PyTorch.”
“Curva de aprendizaje, métricas (Accuracy, Precision, Recall, F1).”
“Historial de predicciones con exportación a CSV.”
“Panel de analítica con tiempo de sesión, visitas por sección y estadísticas históricas.”
“Dataset cargado directamente desde UCI (para EDA) y desde archivo local procesado (para modelo).”

### Funcionalidades

- **Pantalla Principal**: Presentación del proyecto y narrativa del caso de negocio.
- **Análisis de Datos (EDA)**: Exploración del dataset, visualizaciones y matriz de correlación.
- **Predicción IA**: Modelo MLP (Perceptrón Multicapa) con PyTorch para clasificar vinos.
  - Métricas de rendimiento (Accuracy, Precision, Recall, F1-Score)
  - Botón "Aleatorizar Muestra" para cargar datos reales del dataset
  - Historial de predicciones guardado automáticamente
- **Analítica Web**: Panel completo de métricas de uso.
  - Reloj de sesión en tiempo real
  - Tracking de navegación por secciones
  - Historial de vinos clasificados con gráfico interactivo (Plotly)
  - Persistencia de sesiones en archivo JSON
  - Exportación de historial a CSV

## Requisitos

- Python 3.9 o superior
- pip actualizado

## Instalación

1. Clonar el repositorio:

```bash
git clone https://github.com/MrRobot707L/Prueba-2-Herramientas.git
cd Prueba-2-Herramientas
```

2. Crear y activar entorno virtual (recomendado):

```powershell
python -m venv venv 
.\venv\Scripts\Activate.ps1 
```

3. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

## Ejecución

Ejecutar la aplicación Streamlit:

```powershell
streamlit run app.py
```

La aplicación se abrirá automáticamente en el navegador en `http://localhost:8501`

## Estructura del Proyecto

```
Prueba-2-Herramientas/
|-- app.py                 # Punto de entrada de la aplicación
|-- requirements.txt       # Dependencias del proyecto
|-- README.md
|-- ui/
|   |-- __init__.py
|   |-- pantalla_principal.py
|   |-- analisis_datos.py
|   |-- prediccion.py
|   |-- analitica.py       # Módulo de analítica web
|-- data/
|   |-- dataset.csv        # Dataset Wine Quality (UCI)
|   |-- procesado.pkl      # Dataset preprocesado
|   |-- generar_pkl.py     # Script de preprocesamiento
|   |-- analytics_history.json  # Historial de sesiones persistido
|-- docs/
    |-- narrativa.md
    |-- responsabilidades.md
    |-- prompts_ia.md
    |-- instrucciones.md
```

## Limitaciones y mejoras a futuro
Bullets:
“Actualmente solo se usa el dataset de vinos tintos (no blancos).”
“El modelo está entrenado con un único split; no hay validación cruzada.”
“Futuro: probar otros modelos (XGBoost, Random Forest), explicar mejor las predicciones (SHAP/LIME) y desplegar en la nube.”

## Equipo de Desarrollo

- **Antonia Montecinos**: Estructura del proyecto, pantalla principal, redacción y corrección de textos y afincación de detalles
- **Cristian Vergara**: Modelo MLP y módulo de predicción
- **Ignacio Ramirez**: Análisis exploratorio de datos, visualizaciones y módulo de analítica web

## Tecnologías Utilizadas

- Streamlit (frontend)
- Pandas (manipulación de datos)
- PyTorch (red neuronal MLP)
- Scikit-learn (métricas de rendimiento)
- Matplotlib / Seaborn (visualizaciones EDA)
- Plotly (gráficos interactivos)

## Fuente de Datos

Dataset Wine Quality Red del UCI Machine Learning Repository (carga desde fuente externa/cloud).

## Asignatura

INFB6052 - Herramientas para Ciencia de Datos
Universidad Técnica Metropolitana
Segundo Semestre 2025
