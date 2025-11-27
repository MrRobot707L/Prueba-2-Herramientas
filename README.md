# Viña CIA - Sommelier Artificial

**Version Preliminar (50%)**

Aplicacion web interactiva desarrollada en Streamlit para el analisis y prediccion de calidad de vinos utilizando redes neuronales.

## Descripcion del Proyecto

Este proyecto utiliza el dataset Wine Quality del repositorio UCI Machine Learning para construir un "Sommelier Artificial" que clasifica vinos como Premium o Estandar basandose en sus caracteristicas fisico-quimicas.

### Funcionalidades

- **Pantalla Principal**: Presentacion del proyecto y narrativa del caso de negocio.
- **Analisis de Datos (EDA)**: Exploracion del dataset, visualizaciones y matriz de correlacion.
- **Prediccion IA**: Modelo MLP (Perceptron Multicapa) con PyTorch para clasificar vinos.

## Requisitos

- Python 3.9 o superior
- pip actualizado

## Instalacion

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

## Ejecucion

Ejecutar la aplicacion Streamlit:

```powershell
streamlit run app.py
```

La aplicacion se abrira automaticamente en el navegador en `http://localhost:8501`

## Estructura del Proyecto

```
Prueba-2-Herramientas/
|-- app.py                 # Punto de entrada de la aplicacion
|-- requirements.txt       # Dependencias del proyecto
|-- README.md
|-- ui/
|   |-- __init__.py
|   |-- pantalla_principal.py
|   |-- analisis_datos.py
|   |-- prediccion.py
|-- data/
|   |-- dataset.csv        # Dataset Wine Quality (UCI)
|   |-- procesado.pkl      # Dataset preprocesado
|   |-- generar_pkl.py     # Script de preprocesamiento
|-- docs/
    |-- narrativa.md
    |-- responsabilidades.md
    |-- prompts_ia.md
    |-- instrucciones.md
```

## Equipo de Desarrollo

- **Antonia Montecinos**: Estructura del proyecto y pantalla principal
- **Cristian Vergara**: Modelo MLP y modulo de prediccion
- **Ignacio Ramirez**: Analisis exploratorio de datos y visualizaciones

## Tecnologias Utilizadas

- Streamlit (frontend)
- Pandas (manipulacion de datos)
- PyTorch (red neuronal MLP)
- Matplotlib / Seaborn (visualizaciones)

## Asignatura

INFB6052 - Herramientas para Ciencia de Datos
Universidad Tecnica Metropolitana
Segundo Semestre 2025
